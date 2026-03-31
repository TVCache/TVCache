import logging
import time
import asyncio
from tool_schema import Response
import chz
import datasets
import tinker
import torch
from tinker import types
from tinker.types.tensor_data import TensorData
from tinker_cookbook import checkpoint_utils, model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import ml_log
import json
from agent_loop import VideoAgentLoop
import random

from typing import List, Dict

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)
random.seed(8)

@chz.chz
class Config:
    base_url: str | None = None
    log_path: str = "./tests/rebuttal_nocache"
    model_name: str = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    batch_size: int = 4
    group_size: int = 8
    learning_rate: float = 4e-5
    max_length: int = 32768
    lora_rank: int = 32
    save_every: int = 5
    max_tokens: int = 1024
    num_turns: int = 5
    sandbox_base_url: str = "http://localhost:5000"
    epochs: int = 10


def get_reward(response: str, answer: str) -> float:
    return 0.0

def get_prompt(question: str, options: Dict[str, str], video_id: str) -> str:
    with open('./prompt.txt', 'r') as prompt_file:
        prompt_template = prompt_file.read()

        # Format the question with options
        formatted_question = question + "\n\nOptions:\n"
        for key in sorted(options.keys()):
            formatted_question += f"{key}: {options[key]}\n"

        prompt = prompt_template.replace('{QUESTION}', formatted_question)
        prompt += f'\nThe associated video name is {video_id}.mp4.'
        prompt = prompt.replace('{response_schema}', json.dumps(Response.model_json_schema(), indent='\t'))
        return prompt

def get_video_dataset() -> List[dict]:
    json_path = './EgoSchema/processed_videos.json'

    dataset = []

    with open(json_path, 'r') as json_file:
        data = json.load(json_file)
        for point in data:
            question_prompt = get_prompt(point['question'], point['options'], point['video_id'])
            dataset.append({"question": question_prompt, "answer": point["correct_answer_index"]})
    
    return dataset[: 100]

def get_task_id(question: str) -> str:
    lines = question.splitlines()
    for line in lines:
        line = line.strip()
        if line.startswith('The associated video name is'):
            parts = line.split()
            return parts[-1]


async def main(config: Config):
    # Setup logging
    ml_logger = ml_log.setup_logging(
        log_dir=config.log_path,
        wandb_project=None,
        wandb_name=None,
        config=config,
        do_configure_logging_module=True,
    )

    # Get tokenizer and renderer
    tokenizer = get_tokenizer(config.model_name)
    renderer_name = model_info.get_recommended_renderer_name(config.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    logger.info(f"Using renderer: {renderer_name}")

    # Load GSM8K dataset
    logger.info("Loading dataset...")
    dataset = get_video_dataset()

    
    train_dataset = []
    for _ in range(config.epochs):
        random.shuffle(dataset)
        train_dataset.extend(dataset)
        
    n_train_batches = len(train_dataset) // config.batch_size

    # Setup training client
    service_client = tinker.ServiceClient(base_url=config.base_url)

    resume_info = checkpoint_utils.get_last_checkpoint(config.log_path)
    if resume_info:
        training_client = await service_client.create_training_client_from_state_async(
            resume_info["state_path"]
        )
        start_batch = resume_info["batch"]
        logger.info(f"Resuming from batch {start_batch}")
    else:
        training_client = await service_client.create_lora_training_client_async(
            base_model=config.model_name, rank=config.lora_rank
        )
        start_batch = 0

    sampling_params = tinker.types.SamplingParams(
        max_tokens=config.max_tokens,
        stop=renderer.get_stop_sequences(),
    )
    # Optimizer step
    adam_params = types.AdamParams(
        learning_rate=config.learning_rate, beta1=0.9, beta2=0.95, eps=1e-8
    )

    logger.info(f"Training for {n_train_batches} batches")

    #  Main training loop
    batch_idx = start_batch
    next_batch_idx = batch_idx + 1

    while batch_idx < n_train_batches:
        # Setup metrics for logging
        t_start = time.time()
        step = batch_idx
        metrics: dict[str, float] = {
            "progress/batch": batch_idx,
            "optim/lr": config.learning_rate,
            "progress/done_frac": (batch_idx + 1) / n_train_batches,
        }

        # Save checkpoint
        if step % config.save_every == 0 and step > 0:
            await checkpoint_utils.save_checkpoint_async(
                training_client=training_client,
                name=f"{step:06d}",
                log_path=config.log_path,
                kind="state",
                loop_state={"batch": batch_idx},
            )

        # Get training batch and convert to datums online
        batch_start = batch_idx * config.batch_size
        batch_end = min((batch_idx + 1) * config.batch_size, len(train_dataset))
        batch_rows = train_dataset[batch_start: batch_end]

        sampling_path_future = await training_client.save_weights_for_sampler_async(name=f"{step:06d}")
        sampling_path = await sampling_path_future
        sampling_client = await service_client.create_sampling_client_async(model_path=sampling_path.path)

        training_datums: list[types.Datum] = []
        batch_rewards: list[float] = []
        batch_reward_lists: List[List[float]] = []

        rollout_groups: List[List[VideoAgentLoop]] = []

        if next_batch_idx < n_train_batches:
            next_batch_start = next_batch_idx * config.batch_size
            next_batch_end = min((next_batch_idx + 1) * config.batch_size, len(train_dataset))
            next_batch_data = train_dataset[next_batch_start: next_batch_end]

            next_batch_task_ids = [get_task_id(nd['question']) for nd in next_batch_data]
            

        for d_idx, data in enumerate(batch_rows):
            agent_loops: List[VideoAgentLoop] = []

            for i in range(config.group_size):
                
                agent_loop = VideoAgentLoop(
                    training_data_point=data,
                    sampling_client=sampling_client,
                    num_turns=config.num_turns,
                    renderer=renderer,
                    sandbox_base_url=config.sandbox_base_url,
                    task_id=get_task_id(data['question'])
                )

                # Start sandbox with unique ID
                sandbox_id = f"batch_{batch_idx}_data_{batch_rows.index(data)}_rollout_{i}"
                await agent_loop.start_sandbox(sandbox_id)
                agent_loops.append(agent_loop)
            
            rollout_groups.append(agent_loops)
        
        tasks = []

        for agent_loop_list in rollout_groups:
            for agent_loop in agent_loop_list:
                tasks.append(agent_loop.run(sampling_params=sampling_params))

        results = await asyncio.gather(*tasks)
        result_index = 0

        batch_agent_results = []

        for data_idx, data in enumerate(batch_rows):
            data_agent_loops = rollout_groups[data_idx]
            data_agent_loop_results = []

            for _ in range(config.group_size):
                data_agent_loop_results.append(results[result_index])
                result_index += 1

            assert len(data_agent_loops) == len(data_agent_loop_results)

            for agent_loop_obj in data_agent_loops:
                try:
                    await agent_loop_obj.stop_sandbox()
                except Exception as e:
                    print(f'Failed to stop agent loop sandbox')
            
            batch_agent_results.append(
                {
                    'data': data,
                    'agent_loops': data_agent_loops,
                    'results': data_agent_loop_results
                }
            )
        
        # Process each group of rollouts
        for item in batch_agent_results:
            data = item['data']
            agent_loops = item['agent_loops'] # rollouts
            rollout_results = item['results']

            group_rewards: list[float] = []
            group_tokens: list[list[int]] = []
            group_logprobs: list[list[float]] = []
            group_advantage_masks: list[list[float]] = []

            for agent_loop, (all_tokens, all_logprobs, advantage_mask) in zip(agent_loops, rollout_results):

                group_tokens.append(all_tokens)
                group_logprobs.append(all_logprobs)
                group_advantage_masks.append(advantage_mask)

                reward = agent_loop.get_reward()
                group_rewards.append(reward)
            

            mean_reward = sum(group_rewards) / len(group_rewards)
            advantages = [reward - mean_reward for reward in group_rewards]
            batch_rewards.append(mean_reward)
            batch_reward_lists.append(group_rewards)

            # Skip if all advantages are zero
            if all(advantage == 0.0 for advantage in advantages):
                continue

            for tokens, logprobs, advantage_mask, advantage in zip(
                group_tokens, group_logprobs, group_advantage_masks, advantages
            ):
                input_tokens = tokens[:-1]
                input_tokens = [int(token) for token in input_tokens]
                target_tokens = tokens[1:]

                # Apply advantage to masked positions
                all_advantages = [mask * advantage for mask in advantage_mask]

                assert (
                    len(input_tokens)
                    == len(target_tokens)
                    == len(logprobs)
                    == len(all_advantages)
                ), (
                    f"len(input_tokens): {len(input_tokens)}, len(target_tokens): {len(target_tokens)}, "
                    f"len(logprobs): {len(logprobs)}, len(all_advantages): {len(all_advantages)}"
                )

                datum = types.Datum(
                    model_input=types.ModelInput.from_ints(tokens=input_tokens),
                    loss_fn_inputs={
                        "target_tokens": TensorData.from_torch(torch.tensor(target_tokens)),
                        "logprobs": TensorData.from_torch(torch.tensor(logprobs)),
                        "advantages": TensorData.from_torch(torch.tensor(all_advantages)),
                    },
                )
                training_datums.append(datum)

        # Training step
        fwd_bwd_future = await training_client.forward_backward_async(
            training_datums, loss_fn="importance_sampling"
        )
        optim_step_future = await training_client.optim_step_async(adam_params)

        _fwd_bwd_result = await fwd_bwd_future
        _optim_result = await optim_step_future

        # Log metrics[]
        metrics["time/total"] = time.time() - t_start
        metrics["reward/average"] = sum(batch_rewards) / len(batch_rewards)
        metrics["reward/list"] = batch_reward_lists 
        ml_logger.log_metrics(metrics, step=batch_idx)

        batch_idx = next_batch_idx
        next_batch_idx += 1

    await checkpoint_utils.save_checkpoint_async(
        training_client=training_client,
        name="final",
        log_path=config.log_path,
        kind="both",
        loop_state={"batch": n_train_batches},
    )
    ml_logger.close()
    logger.info("Training completed")


if __name__ == "__main__":
    asyncio.run(chz.nested_entrypoint(main))