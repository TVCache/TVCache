import re
import matplotlib.pyplot as plt
import numpy as np
import json
import ast

plt.rcParams.update({'font.size': 14})

def smooth(data, window=3):
    """Compute moving average for smoothing."""
    return np.convolve(data, np.ones(window)/window, mode='valid')

def plot_rewards(qwen_data, cached_data, use_trend=True, window=4, num_epochs=5):
    """Plot rewards with either trend+shading or plain lines."""
    plt.figure(figsize=(5, 3))

    num_steps = len(qwen_data)
    steps_per_epoch = num_steps / num_epochs

    if use_trend:
        qwen_arr = np.array(qwen_data)
        cached_arr = np.array(cached_data)
        qwen_smooth = smooth(qwen_arr, window)
        cached_smooth = smooth(cached_arr, window)
        x_smooth = np.arange(window//2, len(qwen_arr) - window//2 + 1) / steps_per_epoch

        # Trim actual data to match smoothed length
        qwen_trimmed = qwen_arr[window//2 : window//2 + len(qwen_smooth)]
        cached_trimmed = cached_arr[window//2 : window//2 + len(cached_smooth)]

        plt.fill_between(x_smooth, qwen_trimmed, qwen_smooth, alpha=0.2, color='#E74C3C')
        plt.plot(x_smooth, qwen_smooth, linestyle='-', linewidth=3, label='No cache', color='#E74C3C')
        plt.fill_between(x_smooth, cached_trimmed, cached_smooth, alpha=0.2, color='#2E86AB')
        plt.plot(x_smooth, cached_smooth, linestyle='-', linewidth=3, label='Cache', color='#2E86AB')
    else:
        x_epochs = np.arange(len(qwen_data)) / steps_per_epoch
        plt.plot(x_epochs, qwen_data, linestyle='-', linewidth=3, label='No cache', color='#E74C3C')
        plt.plot(x_epochs, cached_data, linestyle='-', linewidth=3, label='Cache', color='#2E86AB')

    plt.xlabel('Epoch')
    plt.ylabel('Mean Reward')
    # plt.legend(frameon=False)
    plt.legend(frameon=False, loc='lower right', bbox_to_anchor=(1.05, -0.05))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('reward_plot_state.pdf')

# Parse qwen.log
rewards_qwen = []
with open('train_tvcache_rebuttal.log', 'r') as f:
    for line in f:
        if 'reward/average' in line:
            # Extract the numeric value using regex
            match = re.search(r'│\s+([\d.]+)\s+│', line)
            if match:
                rewards_qwen.append(float(match.group(1)))

# Parse qwen-cached.log
rewards_cached = []
with open('train_nocache_rebuttal.log', 'r') as f:
    for line in f:
        if 'reward/average' in line:
            # Extract the numeric value using regex
            match = re.search(r'│\s+([\d.]+)\s+│', line)
            if match:
                rewards_cached.append(float(match.group(1)))

# Plot the rewards (first 200 iterations only)
# Set use_trend=True for trend line with shading, False for plain lines
siz = min(len(rewards_qwen), len(rewards_cached))
print(siz)
# siz = 125
plot_rewards(rewards_qwen[: siz], rewards_cached[: siz], use_trend=False)


# with open('train.log', 'r') as log_file:
#     total_cache_hits = 0

#     total_misses = 0
#     useful_hits = 0

#     for line in log_file:
#         line = line.strip()

#         # CACHE HIT, type 1 for task id 93b1de09-2f89-4b4b-94ac-9d2d94df3d66.mp4. with tool calls : ['{"function_name": "load_video_into_sandbox", "argument": "93b1de09-2f89-4b4b-94ac-9d2d94df3d66.mp4"}'] in 0.005073040956631303 seconds and saved tool call time 0.3024447690695524 and depth 1 for rollout 93b1de09-2f89-4b4b-94ac-9d2d94df3d66.mp4.

#         if 'CACHE HIT,' in line:
#             parts = line.split('with tool calls :')
#             data = parts[1]
#             data = data.split("']")[0]
#             data += "']"
#             tool_calls_list = ast.literal_eval(data.strip())

#             parsed_tool_call = json.loads(tool_calls_list[-1])
#             func_name = parsed_tool_call['function_name']
#             if func_name != 'load_video_into_sandbox' and func_name != 'preprocess':
#                 useful_hits += 1
            
#             total_cache_hits += 1

#             if func_name == 'caption_retrieval':
#                 print(line)
        
#         if 'CACHE MISS,' in line:
#             total_misses += 1
    
#     print(f'Total cache hits: {total_cache_hits}, useful cache hits {useful_hits}, misses: {total_misses}')
