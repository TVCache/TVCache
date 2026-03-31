"""Prefix tree (trie) implementation for caching tool call histories."""

from typing import Dict, List, Tuple, Optional, Set
import threading
import os
import time


class Node:
    """A node in the prefix tree."""

    def __init__(self):
        self.children: Dict[str, Node] = {}
        self.env_id = None
        self.value = None
        self.tool_exec_time = None
        self.consumed = False
        self.should_fork = False
        self.consumed_suffix: List[str] = None
        self.init_time = int(time.time()) # Time zone is not necessary because the datastructure is specific to this process
        self.test_result: str = None

    def set_env_id(self, env_id):
        """Set the environment ID for this node."""
        self.env_id = env_id

    def __str__(self):
        """Return a string representation of the node."""
        children_keys = list(self.children.keys())
        return f"Node(env_id={self.env_id}, value={self.value}, tool_exec_time={self.tool_exec_time}, children={children_keys})"

    def __repr__(self):
        """Return a string representation of the node for debugging."""
        return self.__str__()


class MutableEnvPrefixTree:
    """Thread-safe prefix tree cache for storing and retrieving tool call histories."""


    def __init__(self):
        self.prefix_tree_list: Dict[str, Node] = {}
        self.prefix_tree_list_lock = threading.Lock()

        self.prefix_tree_locks: Dict[str, threading.Lock] = {}
        self.locks_lock = threading.Lock()
        self.stored_env_ids = set()
        

        self.cache_budget = int(os.environ.get('CACHE_BUDGET', 50))
        self.referenced_env_ids: Dict[str, int] = {}

        print('CACHE BUDGET={}'.format(self.cache_budget))


    def get_prefix_tree_lock(self, task_name: str) -> threading.Lock:
        """Get or create a lock for the given task_name."""
        if task_name not in self.prefix_tree_locks:
            with self.locks_lock:
                if task_name not in self.prefix_tree_locks:
                    self.prefix_tree_locks[task_name] = threading.Lock()
        return self.prefix_tree_locks[task_name]
    
    def get_root(self, task_name: str) -> Node:
        with self.prefix_tree_list_lock:
            if task_name not in self.prefix_tree_list:
                self.prefix_tree_list[task_name] = Node()

            return self.prefix_tree_list[task_name]


    def get(self, task_name: str, tool_calls: List[str]) -> Tuple[bool, Optional[str], Optional[any], Optional[float]]:
        """Get a cached environment by exact tool call sequence.

        Args:
            task_name: The name of the task
            tool_calls: List of tool calls to match

        Returns:
            Tuple of (found, env_id, value, tool_exec_time)
        """
        found_node = None
        root = self.get_root(task_name)

        if root is not None:
            with self.get_prefix_tree_lock(task_name):
                for i, tool_call in enumerate(tool_calls):
                    if root and tool_call in root.children:
                        root = root.children[tool_call]

                        if i == len(tool_calls) - 1:
                            found_node = root
                            break
                    else:
                        break

        return (
            found_node is not None,
            found_node.env_id if found_node is not None else None,
            found_node.value if found_node is not None else None,
            found_node.tool_exec_time if found_node is not None else None
        )


    def prefix_match(self, task_name: str, tool_calls: List[str]) -> Tuple[bool, Optional[str], Optional[List[str]]]:
        """Find a cached environment with matching tool call prefix.

        Args:
            task_name: The name of the task
            tool_calls: List of tool calls to match against

        Returns:
            Tuple of (found, env_id, history)
        """
        chain: List[Node] = []
        latest_env_len = 0
        history = []

        root = self.get_root(task_name)

        if root is not None:
            with self.get_prefix_tree_lock(task_name):
                for _, tool_call in enumerate(tool_calls):
                    if tool_call in root.children:
                        history.append(tool_call)
                        root = root.children[tool_call]
                        chain.append(root)
                        if root.env_id:
                            latest_env_len = len(chain)
                    else:
                        break

        chain = chain[:latest_env_len]
        history = history[:latest_env_len]

        return (
            len(chain) > 0,
            chain[-1].env_id if len(chain) > 0 else None,
            history if len(history) > 0 else None
        )

    def _prune_cache_helper(self, node: Node, envs: List[Node]):
        if node.env_id and node.env_id not in self.referenced_env_ids:
            envs.append(node)
        for child in node.children:
            self._prune_cache_helper(node.children[child], envs)
    
    def _print_tree_helper(self, node: Node, envs: List[Node]):
        if node.env_id and node.env_id:
            envs.append(node)
        for child in node.children:
            self._print_tree_helper(node.children[child], envs)
    
    def print_tree_nodes(self):
        all_nodes_with_env: List[Node] = []
        with self.prefix_tree_list_lock:
            for task_name in self.prefix_tree_list:
                envs = []
                self._print_tree_helper(self.prefix_tree_list[task_name], envs)
                all_nodes_with_env.extend(envs)
        env_ids = [r.env_id for r in all_nodes_with_env]
        return env_ids
        
    def prune_cache_if_required(self):
        """Prune the cache if it exceeds the budget.

        Uses a policy that prunes environments with the least number of children first,
        as they are less likely to be reused for cache hits.

        Returns:
            List of nodes that were pruned
        """
        all_nodes_with_env: List[Node] = []
        with self.prefix_tree_list_lock:
            for task_name in self.prefix_tree_list:
                envs = []
                self._prune_cache_helper(self.prefix_tree_list[task_name], envs)
                all_nodes_with_env.extend(envs)

        print(f'[BUG]: Number of nodes with enviornment = {len(all_nodes_with_env)}')
        if len(all_nodes_with_env) > self.cache_budget:
            all_nodes_with_env.sort(key=lambda node: (len(node.children), node.init_time))
            num_to_prune = len(all_nodes_with_env) - self.cache_budget
            pruned_nodes = all_nodes_with_env[:num_to_prune]

            removed_envs = [n.env_id for n in pruned_nodes]

            for removed_env in removed_envs:
                self._unmark_env_id_as_present(removed_env)
                print(f'[BUG]: Set after removing {removed_env} is {len(self.stored_env_ids)} | {len(self.print_tree_nodes())}')
            
            # remove the env ids
            for n in pruned_nodes:
                n.env_id = None

            # TODO: Handle if there are not enough envs to remove
            return removed_envs

        return []
        

    def _get_hot_nodes_helper(self, node: Node, hot_nodes: List[Tuple[List[str]]], curent_path: List[str], k: int):
        for child_cmd in node.children:
            child_node = node.children[child_cmd]
            child_path = [x for x in curent_path]
            child_path.append(child_cmd)
            if len(child_node.children) > k:
                hot_nodes.append((child_path))
            self._get_hot_nodes_helper(child_node, hot_nodes, child_path, k)


    
    def get_hot_nodes(self, k: int) -> Dict[str, List[Tuple[List[str]]]]:
        """Return nodes that have more than k children for now, can use any policy"""
        task_hot_nodes = {}
        with self.prefix_tree_list_lock:
            for task_name in self.prefix_tree_list:
                hot_nodes = []
                self._get_hot_nodes_helper(self.prefix_tree_list[task_name], hot_nodes, [], k)
                task_hot_nodes[task_name] = hot_nodes
        
        return task_hot_nodes

    def _mark_env_id_as_present(self, env_id: str):
        with self.prefix_tree_list_lock:
            self.stored_env_ids.add(env_id)
    
    def _unmark_env_id_as_present(self, env_id: str):
        with self.prefix_tree_list_lock:
            self.stored_env_ids.remove(env_id)
        
    def check_env_marked(self, env_id: str):
        with self.prefix_tree_list_lock:
            print(f'STORED ENVS: {self.stored_env_ids}')
            return env_id in self.stored_env_ids

    def put(self, task_name: str, history: List[str], env_id: str, values: List[str] = None, tool_exec_times: List[float] = None, start_idx: int = 0) -> Tuple[bool, List[str]]:
        """Store a tool call history in the cache.

        Args:
            task_name: The name of the task
            history: List of tool calls
            env_id: The environment ID
            value: Optional value to store
            tool_exec_time: Optional execution time for the tool call

        Returns:
            True if successful
        """

        root = self.get_root(task_name)
        print(f'History={history}, values={values}, times={tool_exec_times}, start_idx={start_idx}')
        with self.get_prefix_tree_lock(task_name):
            for idx, tool_call in enumerate(history):
                if tool_call not in root.children:
                    root.children[tool_call] = Node()
                    root.children[tool_call].value = values[idx - start_idx]
                    root.children[tool_call].tool_exec_time = tool_exec_times[idx - start_idx]
                if root.env_id == env_id:
                    root.env_id = None
                root = root.children[tool_call]
            
            if root.env_id != None:
                self._unmark_env_id_as_present(root.env_id)
                print(f'[BUG]: Set after removing {root.env_id} is {len(self.stored_env_ids)} | {len(self.print_tree_nodes())}')

            root.set_env_id(env_id)
            self._mark_env_id_as_present(env_id)
            print(f'[BUG]: Set after adding {root.env_id} is {len(self.stored_env_ids)} | {len(self.print_tree_nodes())}')

        removed = self.prune_cache_if_required()
        return True, removed
    
    def put_immutable(self, task_name: str, history: List[str], env_id: str, values: List[str] = None, tool_exec_times: List[float] = None, start_idx: int = 0) -> Tuple[bool, List[str]]:
        """Store a tool call history in the cache and mark the environment as immutable. Any rollout that needs to extend this node will have to fork and execute

        Args:
            task_name: The name of the task
            history: List of tool calls
            env_id: The environment ID
            value: Optional value to store
            tool_exec_time: Optional execution time for the tool call

        Returns:
            True if successful
        """

        root = self.get_root(task_name)
        print(f'History={history}, values={values}, times={tool_exec_times}, start_idx={start_idx} | IMMUTABLE PUT')
        with self.get_prefix_tree_lock(task_name):
            for idx, tool_call in enumerate(history):
                if tool_call not in root.children:
                    root.children[tool_call] = Node()
                    root.children[tool_call].value = values[idx - start_idx]
                    root.children[tool_call].tool_exec_time = tool_exec_times[idx - start_idx]
                    # This makes the node immutable
                    root.should_fork = True
                    root.consumed = True

                if root.env_id == env_id:
                    root.env_id = None
                root = root.children[tool_call]
            
            removed = []
            if root.env_id != None:
                # If there was a previous environment here, delete it
                with self.prefix_tree_list_lock:
                    # if the env is in referenced list that means some rollout is forking it, wait for it to complete
                    if root.env_id not in self.referenced_env_ids:
                        self.stored_env_ids.remove(root.env_id)
                        removed.append(root.env_id)

            root.set_env_id(env_id)
            self._mark_env_id_as_present(env_id)
            print(f'[BUG]: Set after adding in immutable {root.env_id} is {len(self.stored_env_ids)} | {len(self.print_tree_nodes())}')

        removed.extend(self.prune_cache_if_required())
        return True, removed
    
    def store_test_result(self, task_name: str, history: List[str], test_result: str) -> bool:

        root = self.get_root(task_name)

        with self.get_prefix_tree_lock(task_name):
            for tool_call in history:
                if tool_call in root.children:
                    root = root.children[tool_call]
                    if tool_call == history[-1]:
                        root.test_result = test_result
    
    def get_test_result(self, task_name: str, history: List[str]) -> Tuple[bool, str]:

        root = self.get_root(task_name)

        with self.get_prefix_tree_lock(task_name):
            for tool_call in history:
                if tool_call in root.children:
                    root = root.children[tool_call]
                    if tool_call == history[-1] and root.test_result != None:
                        return True, root.test_result
        
        return False, None
    
    def can_extend(self, task_name: str, history: List[str], suffix: List[str]) -> Tuple[bool, List[str]]:
        """Check if a node can be extended with a suffix and mark it as consumed.

        This method navigates to a node at the specified history path and checks
        if it can be extended with a suffix. If the node hasn't been consumed yet,
        it marks the node as consumed and stores the provided suffix. If already
        consumed, it returns the existing suffix that was previously stored.

        Args:
            task_name: The name of the task
            history: List of tool calls forming the path to the node
            suffix: The suffix to potentially extend with

        Returns:
            Tuple of (can_extend, suffix) where:
            - can_extend: True if the node can be extended (wasn't consumed), False otherwise
            - suffix: The provided suffix if can_extend=True, or the existing suffix if can_extend=False
        """
        root = self.get_root(task_name)

        with self.get_prefix_tree_lock(task_name):
            for command in history:
                if command in root.children:
                    root = root.children[command]
                else:
                    root = None
                    break

            if root and root.should_fork:
                return False, []

            if root and not root.consumed:
                with self.prefix_tree_list_lock:
                    self.ref(env_id=root.env_id)
                    root.consumed_suffix = suffix
                    root.consumed = True
                    return True, suffix
        
            return False, root.consumed_suffix
        
        return False, []

    def ref(self, env_id: str):
        with self.prefix_tree_list_lock:
            if env_id not in self.referenced_env_ids:
                self.referenced_env_ids[env_id] = 0
            self.referenced_env_ids[env_id] += 1

    def unref(self, env_id: str, task_name: str):
        with self.prefix_tree_list_lock:
            if env_id in self.referenced_env_ids:
                self.referenced_env_ids[env_id] -= 1
                if self.referenced_env_ids[env_id] == 0:
                    del self.referenced_env_ids[env_id]

    def should_fork(self, task_name: str, history: List[str]) -> Tuple[bool, str]:
        """Check if a node should be forked based on its consumed status.

        This method navigates to a node at the specified history path and checks
        if it should be forked. A node should be forked if it has been consumed
        and its should_fork flag is set to True.

        Args:
            task_name: The name of the task
            history: List of tool calls forming the path to the node

        Returns:
            Tuple of (should_fork, env_id) where:
            - should_fork: True if the node is consumed and should_fork flag is set, False otherwise
            - env_id: The environment ID of the node if should_fork=True, None otherwise
        """
        root = self.get_root(task_name)

        if root != None:
            with self.get_prefix_tree_lock(task_name):
                for command in history:
                    if command in root.children:
                        root = root.children[command]
                    else:
                        root = None
                        break

                if root and root.consumed and root.should_fork:
                    self.ref(root.env_id)
                    return True, root.env_id
        
        return False, None


    def serialize(self) -> dict:
        """Serialize the entire prefix tree for visualization.

        Returns:
            Dictionary representation of the entire tree
        """
        def serialize_node(node: Node) -> dict:
            """Recursively serialize a node and its children."""
            return {
                "env_id": node.env_id,
                "value": node.value,
                "tool_exec_time": node.tool_exec_time,
                "test_result": node.test_result,
                "children": {key: serialize_node(child) for key, child in node.children.items()}
            }

        with self.prefix_tree_list_lock:
            tree_data = {task_name: serialize_node(root) for task_name, root in self.prefix_tree_list.items()}

        return tree_data
    

    def delete_path(self, task_name: str, history: List[str]) -> bool:
        """Delete a path from the prefix tree and prune childless nodes.

        This method navigates to a node at the specified history path and removes
        it from the tree. It then walks back up the tree, removing any parent nodes
        that have become childless as a result of the deletion.

        Args:
            task_name: The name of the task
            history: List of tool calls forming the path to delete

        Returns:
            True if the path was successfully deleted, False if the task or path doesn't exist
        """
        prefix_tree = None

        with self.prefix_tree_list_lock:
            prefix_tree = self.prefix_tree_list.get(task_name, None)

        if prefix_tree == None:
            # nothing to delete
            return False

        chain: List[Node] = []

        with self.prefix_tree_locks[task_name]:
            for command in history:
                if command in prefix_tree.children:
                    prefix_tree = prefix_tree.children[command]
                    chain.append(prefix_tree)

            if len(chain) != len(history):
                print("Path requested to delete does not exist in the prefix tree")
                return False

            child = None
            parent_child_edge = None
            print(f'Chain: {chain}')
            print(f'History: {history}')
            
            for idx in range(len(chain) - 1, -1, -1):
                if child != None:
                    if len(child.children) == 0:
                        print(f'Deleting {parent_child_edge} from {chain[idx]}')
                        del chain[idx].children[parent_child_edge]

                child = chain[idx]
                parent_child_edge = history[idx]
            
            return True