"""Script to analyze environment lifecycle from test.log."""

def parse_log_file(log_file_path):
    """Parse the log file and count created, deleted, and alive environments.

    Args:
        log_file_path: Path to the log file to analyze.
    """
    created_envs = set()
    deleted_envs = set()

    with open(log_file_path, 'r') as f:
        for line in f:
            line = line.strip()

            # Check for "Creating rollout environment:" pattern
            if "Creating rollout environment:" in line:
                # Split and get the UUID after the colon
                parts = line.split("Creating rollout environment:")
                if len(parts) > 1:
                    env_id = parts[1].strip()
                    created_envs.add(env_id)

            # Check for deletion patterns
            elif "Deleting The created environment" in line or "Deleting The environment" in line:
                # Extract UUID - it's between "environment " and " was" or " of"
                if " was not used by the prefix tree" in line:
                    parts = line.split("environment ")
                    if len(parts) > 1:
                        env_part = parts[1].split(" was")[0].strip()
                        deleted_envs.add(env_part)
                elif " of task " in line:
                    parts = line.split("environment ")
                    if len(parts) > 1:
                        env_part = parts[1].split(" of task")[0].strip()
                        deleted_envs.add(env_part)

    # Calculate alive environments (created but not deleted)
    alive_envs = created_envs - deleted_envs

    # Print results
    print(f"Total Created Environments: {len(created_envs)}")
    print(f"Total Deleted Environments: {len(deleted_envs)}")
    print(f"Total Alive Environments: {len(alive_envs)}")
    print()

    if alive_envs:
        print("Alive Environment IDs:")
        for env_id in sorted(alive_envs):
            print(f"  - {env_id}")
    else:
        print("No alive environments (all created environments were deleted).")


if __name__ == "__main__":
    log_file_path = "test.log"
    parse_log_file(log_file_path)
