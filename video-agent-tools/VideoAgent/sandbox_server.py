from flask import Flask, request, jsonify
from sandbox_manager import SandboxManager
import traceback
import time

app = Flask(__name__)
sandbox_manager = SandboxManager(base_dir='./sandboxes', show_tracking=False)

@app.route('/start', methods=['POST'])
def start_sandbox():
    """Start a sandbox by creating the sandbox directory."""
    try:
        data = request.get_json()
        sandbox_id = data.get('sandbox_id')

        if not sandbox_id:
            return jsonify({'error': 'sandbox_id is required'}), 400

        # Create sandbox
        sandbox_path = sandbox_manager.create_sandbox(sandbox_id)

        return jsonify({
            'success': True,
            'sandbox_id': sandbox_id,
            'sandbox_path': sandbox_path
        }), 200

    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

@app.route('/stop', methods=['POST'])
def stop_sandbox():
    """Stop and remove a sandbox."""
    try:
        data = request.get_json()
        sandbox_id = data.get('sandbox_id')

        if not sandbox_id:
            return jsonify({'error': 'sandbox_id is required'}), 400

        result = sandbox_manager.stop_sandbox(sandbox_id)

        return jsonify({
            'success': result,
            'sandbox_id': sandbox_id
        }), 200

    except Exception as e:
        print(f'Failed to stop sandbox: {traceback.format_exc()}', flush=True)
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

@app.route('/fork', methods=['POST'])
def fork_sandbox():
    """Fork an existing sandbox."""
    try:
        data = request.get_json()
        sandbox_id = data.get('sandbox_id')

        if not sandbox_id:
            return jsonify({'error': 'sandbox_id is required'}), 400

        result = sandbox_manager.fork(sandbox_id)

        return jsonify({
            'sandbox_id': result['sandbox_id']
        }), 200

    except Exception as e:
        print(f'Failed to fork sandbox: {traceback.format_exc()}', flush=True)
        with open('sandbox.log', 'a') as log_file:
            log_file.write(traceback.format_exc())
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


@app.route('/execute', methods=['POST'])
def execute_command():
    """Execute a command in the sandbox."""
    try:
        data = request.get_json()
        sandbox_id = data.get('sandbox_id')
        command = data.get('command')
        argument = data.get('argument', '')

        if not sandbox_id or not command:
            return jsonify({'error': 'sandbox_id and command are required'}), 400

        # Check if sandbox exists
        if not sandbox_manager.sandbox_exists(sandbox_id):
            return jsonify({'error': f'Sandbox {sandbox_id} does not exist'}), 404

        # Execute the appropriate command
        if command == 'load_video_into_sandbox':
            if not argument:
                return jsonify({'error': 'argument (video_name) is required'}), 400
            result = sandbox_manager.load_video_into_sandbox(argument, sandbox_id)

        elif command == 'preprocess':
            sandbox_manager.preprocess(sandbox_id)
            result = 'Preprocessing completed'

        elif command == 'object_memory_querying':
            if not argument:
                return jsonify({'error': 'argument (question) is required'}), 400
            result = sandbox_manager.object_memory_querying(sandbox_id, argument)
            if result == None:
                result = 'I cannot answer that question'

        elif command == 'segment_localization':
            st = time.perf_counter()
            if not argument:
                return jsonify({'error': 'argument (description) is required'}), 400
            result = sandbox_manager.segment_localization(sandbox_id, argument)
            et = time.perf_counter()
            print(f'Time taken for segment localizaiton is: {et - st}')

        elif command == 'caption_retrieval':
            if not argument:
                return jsonify({'error': 'argument (tuple) is required'}), 400
            result = sandbox_manager.caption_retrieval(sandbox_id, argument)

        elif command == 'visual_question_answering':
            if not argument:
                return jsonify({'error': 'argument (tuple) is required'}), 400
            result = sandbox_manager.visual_question_answering(sandbox_id, argument)

        else:
            return jsonify({'error': f'Unknown command: {command}'}), 400

        print(f'Result of calling: {command} is : {result}')
        
        return jsonify({
            'success': True,
            # 'sandbox_id': sandbox_id,
            'command': command,
            'result': result
        }), 200

    except Exception as e:
        print(f'Found error while executing command {traceback.format_exc()}', flush=True)
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
