from typing import Dict, List
from typing import Type
from flask import Flask, request, jsonify
from threading import Lock
from tvclient.utils.tvcache_client import TVCacheClient
from tvclient.tools.tool_call_env import ToolCallEnv
import requests
import time
import threading
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Add file handler for fork cache server logs
fork_cache_file_handler = logging.FileHandler('fork_cache_server.log')
fork_cache_file_handler.setLevel(logging.DEBUG)
fork_cache_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fork_cache_file_handler.setFormatter(fork_cache_formatter)
logger.addHandler(fork_cache_file_handler)

PORT = 4848

class _ForkCacheServer:
    def __init__(self, host: str = '0.0.0.0', port: int = PORT):
        self.app = Flask(__name__)
        self.host = host
        self.port = port
        self.fork_cache: Dict[str, Dict[str, List[str]]] = {}
        self.lock = Lock()
        self._register_routes()
        self.tasks_withdraw_counter = {}

    
    def _register_routes(self):
        @self.app.route('/add', methods=['POST'])
        def add_to_cache():
            data = request.json
            task_name = data.get('task_name')
            parent_env = data.get('parent_env')
            forked_env_ids = data.get('forked_env_ids', [])
            
            with self.lock:
                
                if task_name not in self.tasks_withdraw_counter and parent_env == "root":
                    self.tasks_withdraw_counter[task_name] = 0
                
                if parent_env == "root":
                    self.tasks_withdraw_counter[task_name] = len(forked_env_ids)
                    
                if task_name not in self.fork_cache:
                    self.fork_cache[task_name] = {}
                
                self.fork_cache[task_name][parent_env] = forked_env_ids
                logger.debug(f'Stored {forked_env_ids} for {task_name} and parent {parent_env}')
            
            return jsonify({'status': 'success'}), 200
        
        @self.app.route('/get/<task_name>/<parent_env_id>', methods=['GET'])
        def get_from_cache(task_name, parent_env_id):
            print(f"[FORK BANK] Received get request for task {task_name} and parent {parent_env_id}")
            if parent_env_id == "root":
                print("[FORK BANK] Got request for root for task", task_name)
            
            with self.lock:
               
                env_id = None
                if task_name in self.fork_cache:
                    if parent_env_id in self.fork_cache[task_name]:
                        forked_ids = self.fork_cache[task_name][parent_env_id]
                        if forked_ids:
                            # Pop the first available environment
                            env_id = forked_ids.pop(0)
                            
                            # Clean up empty lists
                            if not forked_ids:
                                del self.fork_cache[task_name][parent_env_id]
                            if not self.fork_cache[task_name]:
                                del self.fork_cache[task_name]
                logger.debug(f"===Addition Fork Cache Stats ===")
                logger.debug(f"Total number of tasks in cache: {len(self.fork_cache)}")
                for task, parents in self.fork_cache.items():
                    total_forked = sum(len(forked) for forked in parents.values())
                    logger.debug(f"  Task '{task}': {len(parents)} parent env(s), {total_forked} forked env(s)")
                logger.debug(f"========================")

                return jsonify({'env_id': env_id}), 200
        
        @self.app.route('/remove/<task_name>', methods=['DELETE'])
        def remove_from_cache(task_name):
            with self.lock:
                if task_name in self.tasks_withdraw_counter:
                    self.tasks_withdraw_counter[task_name] -= 1
                    if self.tasks_withdraw_counter[task_name] <= 1:
                        del self.tasks_withdraw_counter[task_name]
                        logger.debug(f"Removed {task_name} from tasks_withdraw_counter")
                else:
                    env_ids = []
                    if task_name in self.fork_cache:
                        for parent_env, forked_ids in self.fork_cache[task_name].items():
                            env_ids.extend(forked_ids)
                        del self.fork_cache[task_name]
                    logger.debug(f'Removing {env_ids} for task {task_name}')
                    logger.debug(f"===Removal Fork Cache Stats ===")
                    logger.debug(f"Total number of tasks in cache: {len(self.fork_cache)}")
                    for task, parents in self.fork_cache.items():
                        total_forked = sum(len(forked) for forked in parents.values())
                        logger.debug(f"  Task '{task}': {len(parents)} parent env(s), {total_forked} forked env(s)")
                    logger.debug(f"========================")
                    return jsonify({'env_ids': env_ids}), 200
            return jsonify({'env_ids': []}), 200
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            return jsonify({'status': 'healthy'}), 200
    
    def run(self, debug: bool = False):
        self.app.run(host=self.host, port=self.port, debug=debug, threaded=True)


class _ForkCacheClient:
    def __init__(self, server_url: str = f'http://localhost:{PORT}'):
        self.server_url = server_url
    
    def add_to_cache(self, task_name: str, parent_env: str, forked_env_ids: List[str]) -> bool:
        try:
            response = requests.post(
                f'{self.server_url}/add',
                json={
                    'task_name': task_name,
                    'parent_env': parent_env,
                    'forked_env_ids': forked_env_ids
                }
            )
            return response.status_code == 200
        except Exception as e:
            print(f"Error adding to cache: {e}")
            return False
    
    def get_from_cache(self, task_name: str, parent_env_id: str) -> str:
        try:
            response = requests.get(f'{self.server_url}/get/{task_name}/{parent_env_id}')
            if response.status_code == 200:
                return response.json().get('env_id')
            return None
        except Exception as e:
            print(f"Error getting from cache: {e}")
            return None
    
    def remove_from_cache(self, task_name: str) -> List[str]:
        try:
            response = requests.delete(f'{self.server_url}/remove/{task_name}')
            if response.status_code == 200:
                return response.json().get('env_ids', [])
            return []
        except Exception as e:
            print(f"Error removing from cache: {e}")
            return []


class ForkGenerator:

    def __init__(self, env_class: Type[ToolCallEnv], cache_server_port: int = PORT):
        self.forks: Dict[str, List[str]] = {}
        self.forks_locks = Lock()
        self.cache_client = TVCacheClient()
        self.fork_cache_client = _ForkCacheClient(f'http://127.0.0.1:{cache_server_port}')
        self.env_class = env_class
        self.port = cache_server_port
        self._ensure_server_running()
    
    def _ensure_server_running(self):
        retries = 0
        max_retries = 10

        while retries < max_retries:
            try:
                response = requests.get(f'{self.fork_cache_client.server_url}/health', timeout=2)
                if response.status_code == 200:
                    print("Fork cache server is running")
                    return True
            except Exception as e:
                print(f"Server not responding, starting it...")
                
                try:
                    server = _ForkCacheServer(host='0.0.0.0', port=self.port)
                    server_thread = threading.Thread(target=server.run, daemon=True)
                    server_thread.start()
                    
                    time.sleep(2)
                    
                except Exception as start_error:
                    print(f"Failed to start server: {start_error}")
                    time.sleep(1)

    def add_to_fork_cache(self, task_name: str, parent_env: str, forked_env_ids: List[str]):
        self.fork_cache_client.add_to_cache(task_name, parent_env, forked_env_ids)
    
    def _warmup_worker(self, task_name: str, result_list: List, result_lock: Lock):
        """Worker function to create a root environment in a thread."""
        try:
            env_obj = self.env_class(task_name=task_name)
            env_id = env_obj.get_id()
            
            with result_lock:
                result_list.append(env_id)
            
            logger.debug(f'Created root env {env_id} for task {task_name}')
        except Exception as e:
            logger.error(f'Error creating root env: {e}')
    
    def _warmup_root_envs_worker(self, task_name: str, rollout_count: int):
        """Worker function that performs the actual warmup operation in a background thread."""
        root_envs_ids = []
        result_lock = Lock()
        threads = []
        start_time = time.time()
        # Create threads for all root environment creations
        for _ in range(rollout_count):
            thread = threading.Thread(
                target=self._warmup_worker,
                args=(task_name, root_envs_ids, result_lock)
            )
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        self.add_to_fork_cache(task_name, "root", root_envs_ids)
        end_time = time.time()
        logger.debug(f'Warmed up {len(root_envs_ids)} root envs in {end_time - start_time:.2f} seconds for task {task_name}')

    
    
    def _fork_worker(self, task_name: str, parent_env_id: str, result_dict: Dict, result_lock: Lock):
        """Worker function to fork a single environment in a thread."""
        start_time = time.time()
        try:
            env_obj = self.env_class(env_id=parent_env_id, task_name=task_name).fork()
            forked_id = env_obj.get_id()
            
            with result_lock:
                if parent_env_id not in result_dict:
                    result_dict[parent_env_id] = []
                result_dict[parent_env_id].append(forked_id)
            end_time = time.time()
            logger.debug(f'Forked {parent_env_id} -> {forked_id} for task {task_name} in {end_time - start_time:.2f} seconds')
        except Exception as e:
            logger.error(f'Error forking {parent_env_id}: {e}')


    def _deposit_worker(self, task_name: str, rollout_count: int):
        """Worker function that performs the actual deposit operation in a background thread."""
        start_time = time.time()
        logger.debug(f'Starting warmup for task {task_name} with {rollout_count} root envs')
        self._warmup_root_envs_worker(task_name, rollout_count)
        logger.debug(f'Warmup completed for task {task_name}, proceeding to fork existing envs')
        env_ids = self.cache_client.get_all_envs(task_name)
        logger.debug(f"[TOTAL ENVS] in the prefix tree while calling deposit for task {task_name}: {len(env_ids)}")
        forked_ids = {}
        rollout_count = 4
        logger.debug(f'Depositing {rollout_count} forks for task {task_name}')

        for id in env_ids:
            forked_ids[id] = []
            for _ in range(rollout_count):
                env_obj = self.env_class(env_id=id, task_name=task_name).fork()
                forked_ids[id].append(env_obj.get_id())
            self.cache_client.unref(id, task_name)
            
            # Add to fork cache server
            self.add_to_fork_cache(task_name, id, forked_ids[id])
        end_time = time.time()

        
        logger.debug(f'Deposited {rollout_count} forks and warmed up root for task {task_name} in {end_time - start_time:.2f} seconds')


    def deposit(self, task_name: str, rollout_count: int):
        """Non-blocking deposit that starts the deposit operation in a background thread."""
        deposit_thread = threading.Thread(
            target=self._deposit_worker,
            args=(task_name, rollout_count),
            daemon=True
        )
        deposit_thread.start()
        logger.debug(f'Started deposit thread for task {task_name} with {rollout_count} forks')
    

    def rm_from_fork_cache(self, task_name: str):
        env_ids = self.fork_cache_client.remove_from_cache(task_name)
        return env_ids
    

    def _stop_worker(self, env_id: str, task_name: str):
        """Worker function to stop an environment in a thread."""
        try:
            env_obj = self.env_class(env_id=env_id, task_name=task_name)
            env_obj.stop()
            logger.debug(f'Stopped env {env_id} for task {task_name}')
        except Exception as e:
            logger.error(f'Error stopping env {env_id}: {e}')


    def withdraw(self, task_name: str):
        """Non-blocking withdraw that starts the withdraw operation in a background thread."""
        start_time = time.time()
        env_ids = self.rm_from_fork_cache(task_name)

        logger.debug(f'Withdrawing and stopping {len(env_ids)} envs for task {task_name}')
        
        # Create threads for all stop operations
        for env_id in env_ids:
            thread = threading.Thread(
                target=self._stop_worker,
                args=(env_id, task_name)
            )
            thread.start()
        
        end_time = time.time()
        logger.debug(f'Finished withdrawing all envs for task {task_name} in {end_time - start_time:.2f} seconds')


    def get_forked_env(self, task_name: str, parent_env_id: str) -> str:
        logger.debug(f'[FORK-BANK]: Getting forked env for task {task_name} and parent {parent_env_id}')
        return self.fork_cache_client.get_from_cache(task_name, parent_env_id)