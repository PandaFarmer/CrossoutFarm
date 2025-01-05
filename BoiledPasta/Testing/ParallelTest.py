import unittest
from unittest.mock import MagicMock

import sys
sys.path.append('../BoiledPasta')

from BehaviorTrees import Parallel as ParallelNode
from BehaviorTrees import NodeStatus as Status



class TestParallelNode(unittest.TestCase):
    
    def setUp(self):
        """ Set up a basic mock environment for tests. """
        # Mock child nodes
        self.mock_success_node = MagicMock()
        self.mock_failure_node = MagicMock()
        self.mock_running_node = MagicMock()

    def test_all_success(self):
        """ Test the case when all children succeed. """
        # Arrange
        self.mock_success_node.tick.return_value = Status.SUCCESS
        child_nodes = [self.mock_success_node, self.mock_success_node]
        parallel_node = ParallelNode(child_nodes)#, success_threshold=2)
        
        # Act
        result = parallel_node.tick()
        
        # Assert
        self.assertEqual(result, Status.SUCCESS)

    def test_one_failure(self):
        """ Test when one node succeeds and one fails. """
        # Arrange
        self.mock_success_node.tick.return_value = Status.SUCCESS
        self.mock_failure_node.tick.return_value = Status.FAILURE
        child_nodes = [self.mock_success_node, self.mock_failure_node]
        parallel_node = ParallelNode(child_nodes)#, success_threshold=2)
        
        # Act
        result = parallel_node.tick()
        
        # Assert
        self.assertEqual(result, Status.FAILURE)

    def test_all_failure(self):
        """ Test the case when all children fail. """
        # Arrange
        self.mock_failure_node.tick.return_value = Status.FAILURE
        child_nodes = [self.mock_failure_node, self.mock_failure_node]
        parallel_node = ParallelNode(child_nodes)#, success_threshold=2)
        
        # Act
        result = parallel_node.tick()
        
        # Assert
        self.assertEqual(result, Status.FAILURE)

    def test_success_threshold(self):
        """ Test a case where success_threshold is set. """
        # Arrange
        self.mock_success_node.tick.return_value = Status.SUCCESS
        self.mock_failure_node.tick.return_value = Status.FAILURE
        child_nodes = [self.mock_success_node, self.mock_failure_node]
        parallel_node = ParallelNode(child_nodes)#, success_threshold=2)
        
        # Act
        result = parallel_node.tick()
        
        # Assert
        self.assertEqual(result, Status.RUNNING)

    def test_all_running(self):
        """ Test when all child nodes are running. """
        # Arrange
        self.mock_running_node.tick.return_value = Status.RUNNING
        child_nodes = [self.mock_running_node, self.mock_running_node]
        parallel_node = ParallelNode(child_nodes)#, success_threshold=1)
        
        # Act
        result = parallel_node.tick()
        
        # Assert
        self.assertEqual(result, Status.RUNNING)

    def test_mixed_running_and_success(self):
        """ Test when one child is running, and the other is successful. """
        # Arrange
        self.mock_success_node.tick.return_value = Status.SUCCESS
        self.mock_running_node.tick.return_value = Status.RUNNING
        child_nodes = [self.mock_success_node, self.mock_running_node]
        parallel_node = ParallelNode(child_nodes)#, success_threshold=1)
        
        # Act
        result = parallel_node.tick()
        
        # Assert
        self.assertEqual(result, Status.RUNNING)

    def test_mixed_running_and_failure(self):
        """ Test when one child is running, and the other fails. """
        # Arrange
        self.mock_failure_node.tick.return_value = Status.FAILURE
        self.mock_running_node.tick.return_value = Status.RUNNING
        child_nodes = [self.mock_failure_node, self.mock_running_node]
        parallel_node = ParallelNode(child_nodes)#, success_threshold=1)
        
        # Act
        result = parallel_node.tick()
        
        # Assert
        self.assertEqual(result, Status.RUNNING)
    
    def test_success_with_custom_threshold(self):
        """ Test success based on a custom success threshold. """
        # Arrange
        self.mock_success_node.tick.return_value = Status.SUCCESS
        self.mock_failure_node.tick.return_value = Status.FAILURE
        child_nodes = [self.mock_success_node, self.mock_failure_node, self.mock_success_node]
        parallel_node = ParallelNode(child_nodes)#, success_threshold=2)
        
        # Act
        result = parallel_node.tick()
        
        # Assert
        self.assertEqual(result, Status.SUCCESS)

if __name__ == '__main__':
    unittest.main()

