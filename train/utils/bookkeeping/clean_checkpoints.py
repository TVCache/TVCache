import tinker

service_client = tinker.ServiceClient()

rest_client = service_client.create_rest_client()


def purge_run(run_id):
    training_run = rest_client.get_training_run(run_id).result()
    print(f"Training Run: {training_run.training_run_id}, LoRA: {training_run.is_lora}")
    checkpoints = rest_client.list_checkpoints(run_id).result()

    print(f"Found {len(checkpoints.checkpoints)} checkpoints")
    for checkpoint in checkpoints.checkpoints:
        print(f"  {checkpoint.checkpoint_type}: {checkpoint.checkpoint_id}")
        if checkpoint.checkpoint_id != '9b27507c-30bb-55ba-8b6c-9403302b4953:train':
            rest_client.delete_checkpoint(run_id, checkpoint.checkpoint_id)
    

future = rest_client.list_training_runs(limit=50)
response = future.result()
print(f"Found {len(response.training_runs)} training runs")
for run_obj in response.training_runs:
    run_id = run_obj.training_run_id
    purge_run(run_id)

# Get next page
next_page = rest_client.list_training_runs(limit=50, offset=50)

# for run_id in run_ids:
#     training_run = rest_client.get_training_run(run_id).result()
#     print(f"Training Run: {training_run.training_run_id}, LoRA: {training_run.is_lora}")
#     checkpoints = rest_client.list_checkpoints(run_id).result()

#     print(f"Found {len(checkpoints.checkpoints)} checkpoints")
#     for checkpoint in checkpoints.checkpoints:
#         print(f"  {checkpoint.checkpoint_type}: {checkpoint.checkpoint_id}")