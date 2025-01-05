from enum import Enum
from typing import List, Optional
from abc import ABC, abstractmethod

from Logger import Logger

logger = logger = Logger("INFO")

class NodeStatus(Enum):
    """Status enum for behavior tree nodes"""
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    RUNNING = "RUNNING"

class Node(ABC):
    """Abstract base class for all behavior tree nodes"""
    def __init__(self, name: str):
        self.name = name
        self.status = NodeStatus.RUNNING
        self.parent: Optional['Node'] = None
    
    @abstractmethod
    def tick(self) -> NodeStatus:
        """Execute the node's behavior"""
        pass

class Composite(Node):
    """Base class for nodes that can have children"""
    def __init__(self, children: List[Node] = None, name = "CompositeDefaultName"):
        super().__init__(name)
        self.children = children or []
        for child in self.children:
            logger.info(f"adding parent to child: {child}")
            child.parent = self

    def add_child(self, child: Node):
        """Add a child node"""
        self.children.append(child)
        child.parent = self

class Sequence(Composite):
    """Executes children in sequence until one fails"""
    def tick(self) -> NodeStatus:
        for child in self.children:
            status = child.tick()
            if status != NodeStatus.SUCCESS:
                self.status = status
                return status
        self.status = NodeStatus.SUCCESS
        return NodeStatus.SUCCESS

class Selector(Composite):
    """Executes children in sequence until one succeeds"""
    def tick(self) -> NodeStatus:
        for child in self.children:
            status = child.tick()
            if status != NodeStatus.FAILURE:
                self.status = status
                return status
        self.status = NodeStatus.FAILURE
        return NodeStatus.FAILURE

class Parallel(Composite):
    """Executes all children in parallel"""
    def __init__(self, children: List[Node] = None, name: str = "ParallelDefaultName"):
                #  success_threshold: float = 1.0):
        super().__init__(name, children)
        self.initial_child_count = len(children)
        # self.success_threshold = success_threshold

    def tick(self) -> NodeStatus:
        success_count = 0
        running_count = 0
        
        for child in self.children:
            status = child.tick()
            if status == NodeStatus.SUCCESS:
                success_count += 1
            elif status == NodeStatus.RUNNING:
                running_count += 1
            elif status == NodeStatus.FAILURE:
                return NodeStatus.FAILURE
        
        # Calculate success ratio
        total_children = len(self.children)
        if total_children == 0:
            return success_count == self.initial_child_count
        
        # success_ratio = success_count / total_children
        
        # Determine node status
        # if success_count > 0 and success_ratio >= self.success_threshold:
        #     self.status = NodeStatus.SUCCESS
        # elif running_count > 0:
        #     self.status = NodeStatus.RUNNING
        # else:
        #     self.status = NodeStatus.FAILURE

        if running_count > 0:
            self.status = NodeStatus.RUNNING
            
        return self.status

class Action(Node):
    """Leaf node that performs an action"""
    def __init__(self, name: str, action_func):
        super().__init__(name)
        self.action_func = action_func

    def tick(self) -> NodeStatus:
        self.status = self.action_func()
        return self.status

# Example usage:
if __name__ == "__main__":
    # Create some simple actions
    def success_action():
        print("Success action executed")
        return NodeStatus.SUCCESS
    
    def failure_action():
        print("Failure action executed")
        return NodeStatus.FAILURE
    
    def running_action():
        print("Running action executed")
        return NodeStatus.RUNNING
    
    # Create action nodes
    success_node = Action("Success", success_action)
    failure_node = Action("Failure", failure_action)
    running_node = Action("Running", running_action)
    
    # Create a sequence node with two children
    sequence = Sequence("MySequence", [success_node, running_node])
    
    # Create a selector node with two children
    selector = Selector("MySelector", [failure_node, success_node])
    
    # Create a parallel node with three children
    parallel = Parallel("MyParallel", [success_node, failure_node, running_node]) 
                    #    success_threshold=0.5)
    
    # Execute the trees
    print("Sequence result:", sequence.tick())
    print("Selector result:", selector.tick())
    print("Parallel result:", parallel.tick())