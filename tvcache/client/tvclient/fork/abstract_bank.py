from abc import ABC, abstractmethod


class AbstractForkGenerator(ABC):
    """
    Abstract base class for fork generation systems that manage environment forking,
    caching, and lifecycle operations.
    """
    
    @abstractmethod
    def deposit(self, task_name: str, rollout_count: int) -> None:
        """
        Initialize and cache forked environments for a given task.
        
        This method should perform the following operations asynchronously:
        1. Warm up root environments based on rollout_count
        2. Fork existing environments in the cache
        3. Store forked environment references for later retrieval
        
        Args:
            task_name: Identifier for the task
            rollout_count: Number of environments to create and fork count per existing environment
        
        Returns:
            None (operation runs in background thread)
        """
        pass
    
    @abstractmethod
    def withdraw(self, task_name: str) -> None:
        """
        Clean up and stop all cached forked environments for a given task.
        
        This method should perform the following operations asynchronously:
        1. Remove all forked environments from cache
        2. Stop each environment to free resources
        
        Args:
            task_name: Identifier for the task whose environments should be cleaned up
        
        Returns:
            None (operation runs in background thread)
        """
        pass

    @abstractmethod
    def get_forked_env(self, task_name: str, parent_env_id: str) -> str:
        """
        Return one of the pre-existing forks of parent_env_id of task task_id

        This method will return a pre-existing forked environemnt's ID and removes it from the internal cache. This method must not re-use forked environments across invokations.

        Args:
            task_name: Identifier for the task 
            parent_env_id: Identifier of the environment to fork
        
        Returns:
            str (environment ID of the forked environment)
        """