"""TVCacheServer for serving the TVCache HTTP API."""

from flask import Flask, request, jsonify, send_from_directory
from typing import List
import os
import threading
import time
import atexit
from pathlib import Path
import json
from tvcache import MutableEnvPrefixTree
from tvcache import ImmutableEnvPrefixTreeCache
from datetime import datetime

app = Flask(__name__)
# cache = MutableEnvPrefixTree()
cache = ImmutableEnvPrefixTreeCache()

# Auto-save configuration
auto_save_config = {
    'enabled': False,
    'interval': 300,  # Default 5 minutes
    'thread': None,
    'stop_event': None
}


def _save_on_shutdown():
    """Save the tree one final time on shutdown."""
    if auto_save_config['enabled']:
        if auto_save_config['stop_event']:
            auto_save_config['stop_event'].set()
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            base_dir = Path.home() / "tv-cache" / "data" / "runs"
            target_dir = base_dir / "auto-saved"
            target_dir.mkdir(parents=True, exist_ok=True)
            filename = f"visualize_{timestamp}.json"
            with open(target_dir / filename, 'w') as f:
                json.dump(cache.serialize(), f, indent=2)
            print(f"[Auto-save] Final save complete: {filename}")
        except Exception as e:
            print(f"[Auto-save] Shutdown save error: {e}")


atexit.register(_save_on_shutdown)


@app.route('/get', methods=['GET'])
def get_endpoint():
    """Get endpoint for retrieving cached environment by task name.

    Query parameters:
        task_name: The name of the task
        exact: Boolean flag for exact match

    Returns:
        JSON response with:
        - found: boolean indicating if match was found
        - env_id: environment ID if found
    """
    task_name = request.args.get('task_name')
    tool_calls: List[str] = request.args.getlist('tool_calls')

    found, env_id, value, tool_exec_time = cache.get(task_name, tool_calls)

    return jsonify({
        "found": found,
        "env_id": env_id,
        "value": value,
        "tool_exec_time": tool_exec_time
    })


@app.route('/prefix_match', methods=['POST'])
def prefix_match_endpoint():
    """Prefix match endpoint for finding cached environments with matching tool call prefix.

    Request body:
        task_name: The name of the task
        tool_calls: List of tool calls to match against

    Returns:
        JSON response with:
        - found: boolean indicating if prefix match was found
        - env_id: environment ID if found
        - history: serialized history dictionary if found
    """
    data = request.get_json()
    task_name = data.get('task_name')
    tool_calls: List[str] = data.get('tool_calls')

    found, env_id, history = cache.prefix_match(task_name, tool_calls)

    return jsonify({
        "found": found,
        "env_id": env_id,
        "history": history
    })

@app.route('/intel_prefix_match', methods=['POST'])
def intel_prefix_match_endpoint():
    """Prefix match endpoint for finding cached environments with matching tool call prefix.

    Request body:
        task_name: The name of the task
        tool_calls: List of tool calls to match against

    Returns:
        JSON response with:
        - found: boolean indicating if prefix match was found
        - env_id: environment ID if found
        - history: serialized history dictionary if found
    """
    data = request.get_json()
    task_name = data.get('task_name')
    tool_calls: List[str] = data.get('tool_calls')

    found, env_id, history, suffix = cache.intel_prefix_match(task_name, tool_calls)

    return jsonify({
        "found": found,
        "env_id": env_id,
        "history": history,
        "suffix": suffix
    })

@app.route('/mark_stateless', methods=['POST'])
def mark_stateless_endpoint():
    """Mark a cached environment as stateless.

    Request body:
        task_name: The name of the task
        history: The serialized history dictionary
        env_id: The environment ID

    Returns:
        JSON response with success status
    """
    data = request.get_json()
    task_name = data.get('task_name')
    history = data.get('history')
    env_id = data.get('env_id')

    if cache.mark_stateless(task_name, history, env_id):
        return jsonify({
            "success": True
        })
    else:
        return jsonify({
            "success": False
        }), 400

@app.route('/put', methods=['PUT'])
def put_endpoint():
    """Put endpoint for storing tool call history.

    Request body:
        task_name: The name of the task
        history: The serialized history dictionary
        env_id: The environment ID
        values: List of values for each tool call (optional)
        tool_exec_times: List of execution times for each tool call (optional)
        start_idx: Starting index for values/tool_exec_times arrays (optional, default 0)

    Returns:
        JSON response with success status and list of removed environment IDs
    """
    data = request.get_json()
    task_name = data.get('task_name')
    history: List[str] = data.get('history')
    env_id = data.get('env_id')
    values = data.get('values')
    tool_exec_times = data.get('tool_exec_times')
    start_idx = data.get('start_idx', 0)

    try:
        success, removed_env_ids = cache.put(task_name, history, env_id, values, tool_exec_times, start_idx)

        return jsonify({
            "success": success,
            "removed_env_ids": removed_env_ids
        })
    except Exception as e:
        print(f'Failed to put data in: History={history}, values={values}, times={tool_exec_times}, start_idx={start_idx}')
        raise e

@app.route('/remove', methods=['DELETE'])
def remove_endpoint():
    """Delete endpoint for deleting a specific tool call history.

    Request body:
        task_name: The name of the task
        history: The serialized history dictionary

    Returns:
        JSON response with success status
    """
    data = request.get_json()
    task_name = data.get('task_name')
    history: List[str] = data.get('history')

    success = cache.delete_path(task_name, history)

    return jsonify({
        "success": success
    })


@app.route('/can_extend', methods=['POST'])
def can_extend():
    """Check if a node can be extended with a suffix and mark it as consumed.

    This endpoint checks if a node at the given history path can be extended
    with a suffix. If the node hasn't been consumed, it marks it as consumed
    and returns the provided suffix. If already consumed, it returns the
    existing suffix.

    Request body:
        task_name: The name of the task
        history: The list of tool calls leading to the node
        suffix: The suffix to potentially extend with

    Returns:
        JSON response with:
        - can_extend: boolean indicating if the node can be extended
        - suffix: the suffix (either the new one or existing one)
    """
    data = request.get_json()
    task_name = data.get('task_name')
    history: List[str] = data.get('history')
    suffix: List[str] = data.get('suffix')

    can_extend, returned_suffix = cache.can_extend(task_name, history, suffix)

    return jsonify({
        "can_extend": can_extend,
        "suffix": returned_suffix
    })


@app.route('/should_fork', methods=['POST'])
def should_fork():
    """Check if a node should be forked based on its consumed status.

    This endpoint checks if a node at the given history path should be forked
    (i.e., if it has been consumed and should_fork flag is set).

    Request body:
        task_name: The name of the task
        history: The list of tool calls leading to the node

    Returns:
        JSON response with:
        - should_fork: boolean indicating if the node should be forked
        - env_id: the environment ID if should_fork is True, None otherwise
    """
    data = request.get_json()
    task_name = data.get('task_name')
    history: List[str] = data.get('history')

    should_fork, env_id = cache.should_fork(task_name, history)

    return jsonify({
        "should_fork": should_fork,
        "env_id": env_id
    })


@app.route('/visualize', methods=['GET'])
def visualize_tree():
    """Get the entire prefix tree structure for visualization.

    Query parameters:
        path: The path to the visualize.json file (e.g., 'terminal-bench/epoch-5/visualize')

    Returns:
        JSON response with the complete prefix tree structure
    """
    path_param = request.args.get('path', '').strip()
    
    # If path parameter is provided, read from file
    if path_param:
        import json
        from pathlib import Path
        
        # Base directory for runs data
        base_dir = Path.home() / "susRL" / "tv-cache" / "data" / "runs"
        
        # Security: normalize and validate path
        requested_path = Path(path_param)
        
        # Prevent directory traversal attacks
        if '..' in requested_path.parts or requested_path.is_absolute():
            return jsonify({"error": "Invalid path"}), 403
        
        # Construct full path
        full_path = (base_dir / requested_path).with_suffix('.json')
        
        # Ensure the resolved path is still within base_dir
        try:
            full_path = full_path.resolve()
            base_dir_resolved = base_dir.resolve()
            if not str(full_path).startswith(str(base_dir_resolved)):
                return jsonify({"error": "Access denied"}), 403
        except Exception:
            return jsonify({"error": "Invalid path"}), 400
        
        # Check if file exists
        if not full_path.exists():
            return jsonify({"error": f"File not found: {path_param}.json"}), 404
        
        # Read and return JSON file
        try:
            with open(full_path, 'r') as f:
                tree_data = json.load(f)
            return jsonify(tree_data), 200
        except json.JSONDecodeError:
            return jsonify({"error": "Invalid JSON file"}), 500
        except Exception as e:
            return jsonify({"error": f"Error reading file: {str(e)}"}), 500
    
    # Default behavior: return current cache state
    tree_data = cache.serialize()
    return jsonify(tree_data)


@app.route('/get_hot_nodes', methods=['GET'])
def get_hot_nodes_endpoint():
    """Get hot nodes (nodes with more than k children) from the cache.

    Query parameters:
        k: The threshold for number of children (default: 1)

    Returns:
        JSON response with:
        - hot_nodes: Dictionary mapping task names to lists of paths
    """
    k = int(request.args.get('k', 1))
    hot_nodes = cache.get_hot_nodes(k)

    return jsonify({"hot_nodes": hot_nodes})


@app.route('/check_env_marked', methods=['GET'])
def check_env_marked_endpoint():
    """Check if an environment ID is marked as present in the cache.

    Query parameters:
        env_id: The environment ID to check

    Returns:
        JSON response with:
        - marked: boolean indicating if the env_id is marked as present
    """
    env_id = request.args.get('env_id')

    if not env_id:
        return jsonify({"error": "env_id parameter is required"}), 400

    marked = cache.check_env_marked(env_id)

    return jsonify({"marked": marked})


@app.route('/store_test_result', methods=['POST'])
def store_test_result_endpoint():
    """Store a test result for a specific node in the cache.

    Request body:
        task_name: The name of the task
        history: The list of tool calls leading to the node
        test_result: The test result string to store

    Returns:
        JSON response with success status
    """
    data = request.get_json()
    task_name = data.get('task_name')
    history: List[str] = data.get('history')
    test_result = data.get('test_result')

    if not task_name or not history or test_result is None:
        return jsonify({"error": "task_name, history, and test_result are required"}), 400

    success = cache.store_test_result(task_name, history, test_result)

    return jsonify({"success": success})


@app.route('/get_test_result', methods=['GET'])
def get_test_result_endpoint():
    """Get the test result for a specific node in the cache.

    Query parameters:
        task_name: The name of the task
        tool_calls: List of tool calls leading to the node

    Returns:
        JSON response with:
        - found: boolean indicating if the node was found
        - test_result: the test result string if found
    """
    task_name = request.args.get('task_name')
    history: List[str] = request.args.getlist('tool_calls')

    if not task_name or not history:
        return jsonify({"error": "task_name and tool_calls are required"}), 400

    found, test_result = cache.get_test_result(task_name, history)

    return jsonify({
        "found": found,
        "test_result": test_result
    })


@app.route('/unref', methods=['POST'])
def unref_endpoint():
    """Unreference an environment ID from the cache.

    Request body:
        env_id: The environment ID to unreference
        task_name: The name of the task (optional for backward compatibility)

    Returns:
        JSON response with success status
    """
    data = request.get_json()
    env_id = data.get('env_id')
    task_name = data.get('task_name')

    if not env_id:
        return jsonify({"error": "env_id is required"}), 400

    cache.unref(env_id, task_name)

    return jsonify({"success": True})


@app.route('/get_all_envs', methods=['GET'])
def get_all_envs_endpoint():
    """Get all environment IDs in the prefix tree for a given task.

    Query parameters:
        task_name: The name of the task

    Returns:
        JSON response with:
        - env_ids: List of all environment IDs in the task's prefix tree
    """
    task_name = request.args.get('task_name')

    if not task_name:
        return jsonify({"error": "task_name parameter is required"}), 400

    env_ids = cache.get_all_envs(task_name)

    return jsonify({"env_ids": env_ids})


@app.route('/')
def serve_visualizer():
    """Serve the visualizer HTML page."""
    return send_from_directory('static', 'visualizer.html')


@app.route('/visualizer.html')
def serve_visualizer_direct():
    """Serve the visualizer HTML page directly."""
    return send_from_directory('static', 'visualizer.html')


@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files."""
    return send_from_directory('static', filename)


@app.route('/runs')
def serve_runs_page():
    """Serve the runs listing HTML page."""
    return send_from_directory('static', 'runs.html')


@app.route('/api/runs')
def list_runs_api():
    """API endpoint to list all available runs."""
    from pathlib import Path
    import json
    
    base_dir = Path.home() / "susRL" / "tv-cache" / "data" / "runs"
    
    if not base_dir.exists():
        return jsonify({
            "error": f"Runs directory not found: {base_dir}",
            "base_dir": str(base_dir),
            "manual_runs": [],
            "auto_saved_runs": []
        }), 404
    
    # Find all visualize*.json files recursively
    manual_runs = []
    auto_saved_runs = []
    
    for json_file in base_dir.rglob('visualize*.json'):
        # Get relative path from base_dir
        rel_path = json_file.relative_to(base_dir)
        # Remove the .json extension for the path parameter
        path_param = str(rel_path.with_suffix(''))
        
        # Check if this is an auto-saved file
        is_auto_saved = path_param.startswith('auto-saved/')
        
        # Create display name without timestamp
        # e.g., "terminal-bench/epoch-0/visualize_20241111_143025" -> "terminal-bench/epoch-0 (14:30:25)"
        parts = path_param.split('/')
        if len(parts) >= 2 and 'visualize_' in parts[-1]:
            # Extract timestamp from filename
            timestamp_part = parts[-1].split('visualize_')[-1]
            if len(timestamp_part) == 15:  # YYYYMMDD_HHMMSS format
                time_str = f"{timestamp_part[9:11]}:{timestamp_part[11:13]}:{timestamp_part[13:15]}"
                if is_auto_saved:
                    # For auto-saved, just show the time
                    display_name = f"Auto-save ({time_str})"
                else:
                    # For manual saves, show path + time
                    display_name = f"{'/'.join(parts[:-1])} ({time_str})"
            else:
                display_name = '/'.join(parts[:-1]) if not is_auto_saved else path_param
        else:
            display_name = path_param
        
        # Try to get task count from the JSON file
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                task_count = len(data)
        except:
            task_count = '?'
        
        modified_date = datetime.fromtimestamp(json_file.stat().st_mtime).strftime('%Y-%m-%d %H:%M')
        
        run_info = {
            'path': path_param,
            'display_name': display_name,
            'task_count': task_count,
            'file_size': json_file.stat().st_size,
            'modified': json_file.stat().st_mtime,
            'modified_date': modified_date
        }
        
        # Add to appropriate list
        if is_auto_saved:
            auto_saved_runs.append(run_info)
        else:
            manual_runs.append(run_info)
    
    # Sort both lists by modified time (newest first)
    manual_runs.sort(key=lambda x: x['modified'], reverse=True)
    auto_saved_runs.sort(key=lambda x: x['modified'], reverse=True)
    
    return jsonify({
        "manual_runs": manual_runs,
        "auto_saved_runs": auto_saved_runs,
        "base_dir": str(base_dir)
    })

@app.route('/api/save', methods=['POST'])
def save_tree_api():
    """API endpoint to save tree data to a file."""
    data = request.get_json()
    run_name = data.get('run_name', '').strip()
    epoch = data.get('epoch')
    tree_data = data.get('data')
    
    if not run_name or epoch is None or not tree_data:
        return jsonify({"error": "run_name, epoch, and data are required"}), 400
    
    # Sanitize run_name to prevent directory traversal
    run_name = run_name.replace('..', '').replace('/', '-').replace('\\', '-')
    
    # Base directory for runs data
    base_dir = Path.home() / "susRL" / "tv-cache" / "data" / "runs"
    
    # Create directory structure: runs/{run_name}/epoch-{epoch}/
    target_dir = base_dir / run_name / f"epoch-{epoch}"
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the file with timestamp to allow multiple saves
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    target_file = target_dir / f"visualize_{timestamp}.json"

    try:
        with open(target_file, 'w') as f:
            json.dump(tree_data, f, indent=2)
        
        # Create relative path for viewing (without timestamp in display)
        relative_path = f"{run_name}/epoch-{epoch}/visualize_{timestamp}"
        
        return jsonify({
            "success": True,
            "path": str(target_file),
            "relative_path": relative_path
        }), 200
    except Exception as e:
        return jsonify({"error": f"Failed to save file: {str(e)}"}), 500

@app.route('/api/delete_run', methods=['DELETE'])
def delete_run_api():
    """API endpoint to delete a run directory."""
    from pathlib import Path
    import shutil
    
    data = request.get_json()
    path_param = data.get('path', '').strip()
    
    if not path_param:
        return jsonify({"error": "path is required"}), 400
    
    # Base directory for runs data
    base_dir = Path.home() / "susRL" / "tv-cache" / "data" / "runs"
    
    # Security: normalize and validate path
    requested_path = Path(path_param)
    
    # Prevent directory traversal attacks
    if '..' in requested_path.parts or requested_path.is_absolute():
        return jsonify({"error": "Invalid path"}), 403
    
    # Get the parent directory (e.g., terminal-bench/epoch-5 -> terminal-bench/epoch-5)
    # We want to delete the epoch directory
    target_path = base_dir / requested_path.parent
    
    # Ensure the resolved path is still within base_dir
    try:
        target_path = target_path.resolve()
        base_dir_resolved = base_dir.resolve()
        if not str(target_path).startswith(str(base_dir_resolved)):
            return jsonify({"error": "Access denied"}), 403
    except Exception:
        return jsonify({"error": "Invalid path"}), 400
    
    # Check if directory exists
    if not target_path.exists():
        return jsonify({"error": f"Directory not found: {path_param}"}), 404
    
    # Delete the directory
    try:
        shutil.rmtree(target_path)
        
        # Check if parent directory is now empty and delete it too
        parent_dir = target_path.parent
        if parent_dir != base_dir and parent_dir.exists():
            try:
                # Only delete if empty
                if not any(parent_dir.iterdir()):
                    parent_dir.rmdir()
            except:
                pass  # Ignore errors when cleaning up parent
        
        return jsonify({
            "success": True,
            "message": f"Successfully deleted {path_param}"
        }), 200
    except Exception as e:
        return jsonify({"error": f"Failed to delete: {str(e)}"}), 500


def _auto_save_worker():
    """Background worker that periodically saves the tree to disk."""
    while not auto_save_config['stop_event'].wait(timeout=auto_save_config['interval']):
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            base_dir = Path.home() / "susRL" / "tv-cache" / "data" / "runs"
            target_dir = base_dir / "auto-saved"
            target_dir.mkdir(parents=True, exist_ok=True)
            target_file = target_dir / f"visualize_{timestamp}.json"
            
            with open(target_file, 'w') as f:
                json.dump(cache.serialize(), f, indent=2)
            
            print(f"[Auto-save] Saved to {target_file}")
        except Exception as e:
            print(f"[Auto-save] Error: {e}")


def _start_auto_save():
    """Start the auto-save background thread."""
    if auto_save_config['enabled'] and auto_save_config['thread'] is None:
        auto_save_config['stop_event'] = threading.Event()
        auto_save_config['thread'] = threading.Thread(target=_auto_save_worker, daemon=True)
        auto_save_config['thread'].start()


def run_server(host: str = '0.0.0.0', port: int = 8000, debug: bool = False,
               auto_save: bool = False, save_interval: int = 300):
    """Run the TVCache server.

    Args:
        host: The host address to bind to
        port: The port to listen on
        debug: Whether to run in debug mode
        auto_save: Whether to enable automatic periodic saving of the tree
        save_interval: The interval in seconds between auto-saves (default: 300 = 5 minutes)
    """
    # Configure auto-save
    if auto_save:
        auto_save_config['enabled'] = True
        auto_save_config['interval'] = save_interval
        
        # Start the auto-save thread
        _start_auto_save()
    
    app.run(host=host, port=port, debug=debug, threaded=True)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run the TVCache server')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Host address to bind to (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=8001,
                        help='Port to listen on (default: 8001)')
    parser.add_argument('--debug', action='store_true',
                        help='Run in debug mode')
    parser.add_argument('--auto-save', action='store_true',
                        help='Enable automatic periodic saving of the tree to auto-saved folder')
    parser.add_argument('--save-interval', type=int, default=3000,
                        help='Interval in seconds between auto-saves (default: 3000)')

    args = parser.parse_args()
    
    run_server(
        host=args.host,
        port=args.port,
        debug=args.debug,
        auto_save=args.auto_save,
        save_interval=args.save_interval
    )
