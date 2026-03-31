from tvcache.prefix_tree import PrefixTree
from typing import Dict, List, Tuple
import time
import threading
import os

class StatelessCommands:
    "Class to handle stateless commands"
    def __init__(self, task_name: str):
        self.stateless_commands = set()
        self.task_name = task_name
        self.last_stateful_command = None

    def add_command(self, command: str):
        self.stateless_commands.add(command)

    def is_stateless(self, command: str) -> bool:
        return command in self.stateless_commands

    def remove_command(self, command: str):
        self.stateless_commands.discard(command)
    
    
class Node:
    """A node in the prefix tree."""

    def __init__(self):
        self.children: Dict[str, Node] = {}
        self.env_id = None
        self.value = None
        self.tool_exec_time = None
        self.init_time = int(time.time()) # Need this for time based eviction
        self.ttl: float = float('inf')
        self.test_result: str = None
        self.cache_hits: int = 0
        self.prefix_hits: int = 0
        self.stateless: bool = False

    def is_expired(self) -> bool:
        return time.time() >= self.init_time + self.ttl

    def set_env_id(self, env_id):
        """Set the environment ID for this node."""
        self.env_id = env_id

    def set_stateless(self):
        """Set the stateless flag for this node."""
        self.stateless = True

    def __str__(self):
        """Return a string representation of the node."""
        children_keys = list(self.children.keys())
        return f"Node(env_id={self.env_id}, value={self.value}, tool_exec_time={self.tool_exec_time}, children={children_keys})"

    def __repr__(self):
        """Return a string representation of the node for debugging."""
        return self.__str__()


class ImmutableEnvPrefixTreeCache(PrefixTree):

    def __init__(self):
        self.prefix_trees_dict_lock: threading.Lock = threading.Lock()

        self.prefix_trees: Dict[str, Node] = {}
        self.prefix_tree_locks: Dict[str, threading.Lock] = {}
        self.prefix_tree_env_refs: Dict[str, Dict[str, int]] = {}
        self.prefix_tree_env_count: Dict[str, int] = {}

        self.cache_budget = int(os.environ.get('CACHE_BUDGET', 50))
        self.watermark = int(1 * self.cache_budget) # Evict after reaching the watermark
        self.safemark = int(0.7 * self.cache_budget) # Evict till safe mark is reached

        self.ttl_cleanup_interval = int(os.environ.get('TTL_CLEANUP_INTERVAL', 30))
        self.ttl_cleanup_stop_event = threading.Event()
        self.ttl_cleanup_thread = threading.Thread(target=self._ttl_cleanup_worker, daemon=True)
        self.ttl_cleanup_thread.start()

    # def set_node_stateless(Node):
    def get_root(self, task_name: str) -> Tuple[Node, threading.Lock]:
        with self.prefix_trees_dict_lock:
            if task_name not in self.prefix_trees:
                self.prefix_trees[task_name] = Node()
                self.prefix_tree_locks[task_name] = threading.Lock()
                self.prefix_tree_env_refs[task_name] = {}
                self.prefix_tree_env_count[task_name] = 0

            return self.prefix_trees[task_name], self.prefix_tree_locks[task_name]

    def _collect_subtree_env_ids(self, node: Node) -> List[str]:
        env_ids = []
        def traverse(n: Node):
            if n.env_id is not None:
                env_ids.append(n.env_id)
            for child in n.children.values():
                traverse(child)
        traverse(node)
        return env_ids

    def _expire_child_if_needed(self, parent: Node, child_key: str, task_name: str):
        child = parent.children[child_key]
        if not child.is_expired():
            return False, []
        removed = self._collect_subtree_env_ids(child)
        self.prefix_tree_env_count[task_name] -= len(removed)
        del parent.children[child_key]
        return True, removed

    def get(self, task_name, tool_calls):
        
        found_node = None
        current_tree_node, tree_lock = self.get_root(task_name)

        with tree_lock:
            for i, tool_call in enumerate(tool_calls):
                if current_tree_node and tool_call in current_tree_node.children:
                    expired, _ = self._expire_child_if_needed(current_tree_node, tool_call, task_name)
                    if expired:
                        break
                    current_tree_node = current_tree_node.children[tool_call]

                    if i == len(tool_calls) - 1:
                        found_node = current_tree_node
                        found_node.cache_hits += 1
                        break
                else:
                    break

        return (
            found_node is not None,
            found_node.env_id if found_node is not None else None,
            found_node.value if found_node is not None else None,
            found_node.tool_exec_time if found_node is not None else None
        )

    def prefix_match(self, task_name, tool_calls):
        chain: List[Node] = []
        latest_env_len = 0
        history = []

        current_tree_node, tree_lock = self.get_root(task_name)

        with tree_lock:
            for _, tool_call in enumerate(tool_calls):
                if tool_call in current_tree_node.children:
                    expired, _ = self._expire_child_if_needed(current_tree_node, tool_call, task_name)
                    if expired:
                        break
                    history.append(tool_call)
                    current_tree_node = current_tree_node.children[tool_call]
                    chain.append(current_tree_node)
                    if current_tree_node.env_id:
                        latest_env_len = len(chain)
                else:
                    break
            
            if len(chain) > 0:
                chain[-1].prefix_hits += 1

            chain = chain[:latest_env_len]
            history = history[:latest_env_len]
            if len(chain) > 0:
                current_tree_node = chain[-1]
            
                if current_tree_node.env_id not in self.prefix_tree_env_refs[task_name]:
                    self.prefix_tree_env_refs[task_name][current_tree_node.env_id] = 0
            
                self.prefix_tree_env_refs[task_name][current_tree_node.env_id] += 1

            return (
                len(chain) > 0,
                chain[-1].env_id if len(chain) > 0 else None,
                history if len(history) > 0 else None
            )

    def sintel_prefix_match(self, task_name, tool_calls):
        raise NotImplementedError

    def intel_prefix_match(self, task_name, tool_calls):
        raise NotImplementedError

    def _prune_prefix_tree_unlocked(self, root: Node) -> List[Node]:
        """Goes over the prefix tree and removes environments in the prefix tree according to the cache eviction policy. Returns a sorted list of Environments that the caller can evict, ordered by least priority to highest priority."""
        all_nodes = []
        
        def traverse(n: Node):
            if n.env_id:
                all_nodes.append(n)
            
            for child in n.children:
                traverse(n.children[child])
        
        traverse(root)

        all_nodes.sort(key=lambda node: (len(node.children), node.init_time))
    
        return all_nodes

    def mark_stateless(self, task_name, history, env_id):
        current_tree_node, tree_lock = self.get_root(task_name)
        with tree_lock:
            for tool_call in history:
                if tool_call in current_tree_node.children:
                    current_tree_node = current_tree_node.children[tool_call]
                    if tool_call == history[-1] and current_tree_node.env_id == env_id:
                       current_tree_node.set_stateless()
                       return True
                else:
                    print(f'Cannot update state, tool call {tool_call} not found in tree')
                    break
        return False


    def put(self, task_name, history, env_id, values = None, tool_exec_times = None, start_idx = 0, ttl = None):
        current_tree_node, tree_lock = self.get_root(task_name)
        
        # Initialize timing variables
        
        with tree_lock:
            
            # Time tree traversal and insertion
            start_node = current_tree_node
            removed = []

            for idx, tool_call in enumerate(history):
                if tool_call not in current_tree_node.children:
                    current_tree_node.children[tool_call] = Node()
                if idx - start_idx >= 0:
                    current_tree_node.children[tool_call].value = values[idx - start_idx]
                    current_tree_node.children[tool_call].tool_exec_time = tool_exec_times[idx - start_idx]
                if current_tree_node.env_id == env_id:
                    current_tree_node.env_id = None
                current_tree_node = current_tree_node.children[tool_call]
            

            # Time env_id cleanup
            if current_tree_node.env_id != None:
                # self.prefix_tree_env_count[task_name] -= 1
                removed.append(env_id)
            else:
                current_tree_node.set_env_id(env_id)
                if ttl is not None:
                    current_tree_node.ttl = ttl

                if env_id != None:
                    self.prefix_tree_env_count[task_name] += 1
                    # print(f'[ENV]: Added {self.prefix_tree_env_count[task_name]}')
            

            # print(f'Total number of environments: {self.prefix_tree_env_count[task_name]}, {task_name}, {self.watermark}, {self.safemark}')

            # Time pruning

            if self.prefix_tree_env_count[task_name] > self.watermark:
                candidates = self._prune_prefix_tree_unlocked(start_node)

                for candidate in candidates:
                    if candidate.env_id in self.prefix_tree_env_refs[task_name] and self.prefix_tree_env_refs[task_name][candidate.env_id] > 0:
                        continue

                    removed.append(candidate.env_id)
                    candidate.env_id = None
                    self.prefix_tree_env_count[task_name] -= 1

                    if self.prefix_tree_env_count[task_name] == self.safemark:
                        break
        
        return True, removed

    def delete_path(self, task_name, history):
        raise NotImplementedError

    def can_extend(self, task_name, history, suffix):
        raise NotImplementedError

    def should_fork(self, task_name, history):
        current_tree_node, tree_lock = self.get_root(task_name)

        with tree_lock:
            for idx, tool_call in enumerate(history):
                if tool_call in current_tree_node.children:
                    current_tree_node = current_tree_node.children[tool_call]

                    # Last tool call, return the env
                    if idx == len(history) - 1:
                        # reference the node
                        if current_tree_node.env_id not in self.prefix_tree_env_refs[task_name]:
                            self.prefix_tree_env_refs[task_name][current_tree_node.env_id] = 0
                        
                        self.prefix_tree_env_refs[task_name][current_tree_node.env_id] += 1
                        return True, current_tree_node.env_id
                else:
                    break    

        return False, None    

    def serialize(self):
        def serialize_node(node: Node) -> dict:
            """Recursively serialize a node and its children."""
            return {
                "env_id": node.env_id,
                "value": node.value,
                "tool_exec_time": node.tool_exec_time,
                "init_time": node.init_time,
                "ttl": node.ttl if node.ttl != float('inf') else None,
                "test_result": node.test_result,
                "cache_hits": node.cache_hits,
                "prefix_hits": node.prefix_hits,
                "stateless": node.stateless,
                "children": {key: serialize_node(child) for key, child in node.children.items()}
            }

        with self.prefix_trees_dict_lock:
            tree_data = {task_name: serialize_node(root) for task_name, root in self.prefix_trees.items()}

        return tree_data

    def get_hot_nodes(self, k = 1):
        raise NotImplementedError

    def check_env_marked(self, env_id):
        raise NotImplementedError

    def store_test_result(self, task_name, history, test_result) -> bool:
        current_tree_node, tree_lock = self.get_root(task_name)

        with tree_lock:
            for tool_call in history:
                if tool_call in current_tree_node.children:
                    current_tree_node = current_tree_node.children[tool_call]
                    if tool_call == history[-1]:
                        current_tree_node.test_result = test_result

    def get_test_result(self, task_name, history) -> Tuple[bool, str]:
        current_tree_lock, tree_lock = self.get_root(task_name)

        with tree_lock:
            for tool_call in history:
                if tool_call in current_tree_lock.children:
                    current_tree_lock = current_tree_lock.children[tool_call]
                    if tool_call == history[-1] and current_tree_lock.test_result != None:
                        return True, current_tree_lock.test_result
        
        return False, None
    
    def unref(self, env_id, task_name=None):
        current_tree_node, tree_lock = self.get_root(task_name)

        with tree_lock:
            self.prefix_tree_env_refs[task_name][env_id] -= 1

    def get_all_envs(self, task_name: str) -> List[str]:
        """Get all environment IDs in the prefix tree for a given task.

        Args:
            task_name: The name of the task

        Returns:
            List of all environment IDs present in the task's prefix tree
        """
        current_tree_node, tree_lock = self.get_root(task_name)
        env_ids = []

        def collect_env_ids(node: Node):
            if node.env_id is not None:
                env_ids.append(node.env_id)
            for child in node.children.values():
                collect_env_ids(child)

        with tree_lock:
            collect_env_ids(current_tree_node)
            for env_id in env_ids:
                if current_tree_node.env_id not in self.prefix_tree_env_refs[task_name]:
                    self.prefix_tree_env_refs[task_name][env_id] = 0
                
                self.prefix_tree_env_refs[task_name][env_id] += 1

        return env_ids

    def _ttl_cleanup_worker(self):
        while not self.ttl_cleanup_stop_event.wait(timeout=self.ttl_cleanup_interval):
            self._run_ttl_cleanup()

    def _run_ttl_cleanup(self) -> Dict[str, List[str]]:
        with self.prefix_trees_dict_lock:
            tasks = [(name, root, self.prefix_tree_locks[name]) for name, root in self.prefix_trees.items()]

        all_removed = {}
        for task_name, root, lock in tasks:
            removed = []
            with lock:
                self._expire_subtrees(root, task_name, removed)
                if len(removed) > 0:
                    all_removed[task_name] = removed
        return all_removed

    def _expire_subtrees(self, node: Node, task_name: str, removed: List[str]) -> List[str]:
        for child_key in list(node.children.keys()):
            expired, removed_node_ids = self._expire_child_if_needed(node, child_key, task_name)
            if expired:
                removed.extend(removed_node_ids)
            else:
                self._expire_subtrees(node.children[child_key], task_name, removed)
