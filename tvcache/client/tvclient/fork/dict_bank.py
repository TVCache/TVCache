from tvclient.utils.async_tvcache_client import AsyncTVCacheClient
from tvclient.tools.tool_call_env import ToolCallEnv
from typing import Type, Dict, List
import logging
from tvclient.fork.abstract_bank import AbstractForkGenerator
from threading import Lock

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

fork_cache_file_handler = logging.FileHandler('fork_cache_server.log')
fork_cache_file_handler.setLevel(logging.DEBUG)
fork_cache_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fork_cache_file_handler.setFormatter(fork_cache_formatter)

logger.addHandler(fork_cache_file_handler)

ROOT_KEY = "root"

class SimpleDictBank(AbstractForkGenerator):

    lock = Lock()
    cache_client = AsyncTVCacheClient()
    warmed_environments: Dict[str, Dict[str, List[str]]] = {}

    def __init__(self, env_class: Type[ToolCallEnv]):
        self.env_class = env_class
        logger.info(f"SimpleDictBank initialized with env_class: {env_class.__name__}")

    
    async def deposit(self, task_name, rollout_count):
        logger.info(f"Starting deposit for task_name='{task_name}', rollout_count={rollout_count}")
        
        stored_env_ids = await self.cache_client.get_all_envs(task_name=task_name)
        logger.debug(f"Retrieved {len(stored_env_ids)} stored environment IDs: {stored_env_ids}")

        forks: Dict[str, List[str]] = {}

        for env_id in stored_env_ids:
            logger.debug(f"Creating forks for env_id='{env_id}'")
            env_obj = self.env_class(env_id, task_name=task_name)
            forks[env_id] = []

            for i in range(rollout_count):
                fork_obj = await env_obj.fork()
                forks[env_id].append(fork_obj.get_id())
                logger.debug(f"Created fork {i+1}/{rollout_count} for env_id='{env_id}': {fork_obj}")
            
            logger.info(f"Completed {rollout_count} forks for env_id='{env_id}'")
        
        # prime the root env 
        logger.debug(f"Creating root environment forks")
        forks[ROOT_KEY] = []

        for i in range(rollout_count):
            root_obj = self.env_class(task_name=task_name)
            forks[ROOT_KEY].append(root_obj.get_id())
            logger.debug(f"Created root fork {i+1}/{rollout_count}: {root_obj.get_id()}")
        
        logger.info(f"Completed {rollout_count} root forks")

        with self.lock:
            if task_name not in self.warmed_environments:
                self.warmed_environments[task_name] = {}
                logger.debug(f"Initialized warmed_environments for task_name='{task_name}'")
            
            task_environments = self.warmed_environments[task_name]

            for env_id in forks:
                if env_id not in task_environments:
                    task_environments[env_id] = []
                    logger.debug(f"Initialized fork list for env_id='{env_id}' in task '{task_name}'")
                
                before_count = len(task_environments[env_id])
                task_environments[env_id].extend(forks[env_id])
                after_count = len(task_environments[env_id])
                logger.info(f"Added {len(forks[env_id])} forks to env_id='{env_id}' (before: {before_count}, after: {after_count})")
            
            total_forks = sum(len(envs) for envs in task_environments.values())
            logger.info(f"Deposit complete for task '{task_name}': {len(task_environments)} parent envs, {total_forks} total forks")
            logger.debug(f"Warmed environments breakdown: {[(env_id, len(envs)) for env_id, envs in task_environments.items()]}")

        for env_id in stored_env_ids:
            logger.debug(f"Unreferencing env_id='{env_id}' for task '{task_name}'")
            await self.cache_client.unref(env_id, task_name=task_name)
        
        logger.info(f"Deposit completed successfully for task_name='{task_name}'")


    async def withdraw(self, task_name):
        logger.info(f"Starting withdraw for task_name='{task_name}'")
        task_environments = {}

        with self.lock:
            if task_name not in self.warmed_environments:
                logger.warning(f"Task '{task_name}' not found in warmed_environments during withdraw")
                return
            
            task_environments = self.warmed_environments[task_name]
            parent_env_count = len(task_environments)
            total_fork_count = sum(len(envs) for envs in task_environments.values())
            logger.info(f"Withdrawing {parent_env_count} parent envs with {total_fork_count} total forks for task '{task_name}'")
            
            self.warmed_environments[task_name] = {}
            logger.debug(f"Cleared warmed_environments for task_name='{task_name}'")
        
        for parent_env in task_environments:
            forked_envs = task_environments[parent_env]
            logger.debug(f"Stopping {len(forked_envs)} forked environments for parent_env='{parent_env}'")
            
            for idx, f_env_id in enumerate(forked_envs):
                logger.debug(f"Stopping fork {idx+1}/{len(forked_envs)}: {f_env_id}")
                f_env_obj = self.env_class(env_id=f_env_id, task_name=task_name)
                await f_env_obj.stop()
            
            logger.info(f"Stopped all {len(forked_envs)} forks for parent_env='{parent_env}'")
        
        logger.info(f"Withdraw completed for task_name='{task_name}'")
    
    def get_forked_env(self, task_name, parent_env_id) -> str:
        logger.debug(f"get_forked_env called for task_name='{task_name}', parent_env_id='{parent_env_id}'")
        
        with self.lock:
            if task_name not in self.warmed_environments:
                logger.warning(f"Task '{task_name}' not found in warmed_environments")
                return None
            
            task_environments = self.warmed_environments[task_name]

            if len(task_environments) == 0:
                logger.warning(f"No warmed environments available for task '{task_name}'")
                return None
            
            if parent_env_id not in task_environments:
                logger.warning(f"Parent env_id '{parent_env_id}' not found in task '{task_name}'. Available: {list(task_environments.keys())}")
                return None
            
            if len(task_environments[parent_env_id]) == 0:
                logger.warning(f"No forks available for parent_env_id '{parent_env_id}' in task '{task_name}'")
                return None
            
            forked_env = task_environments[parent_env_id].pop()
            remaining = len(task_environments[parent_env_id])
            logger.info(f"Retrieved fork '{forked_env}' from parent '{parent_env_id}' (remaining: {remaining})")
            
            return forked_env
        
        return None