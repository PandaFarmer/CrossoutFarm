import py_trees
from py_trees.composites import Selector, Sequence, Parallel
from py_trees.behaviour import Behaviour
import functools
import logging

logger = logging.getLogger(__name__)

class Mapping(Behaviour):
    def __init__(self, name):
        super(Mapping, self).__init__(name)
        self.hasrun = False
        self.name = name
        
        
    def update(self):
        if self.hasrun:
            return py_trees.common.Status.SUCCESS
        if self.name == 'map the environment':
            return self.MapTheEnvironment()

        return py_trees.common.Status.RUNNING

        
    def MapTheEnvironment(self):
        logger.info('MappingTheEnvironment')
        return py_trees.common.Status.SUCCESS
        
class Navigation(Behaviour):
    def __init__(self, name):
        super(Navigation, self).__init__(name)
        self.hasrun = False
        self.name = name
        
    def update(self):
        if self.hasrun:
            return py_trees.common.Status.SUCCESS
        if self.name == 'simulate movement':
            return self.SimulateMovement()

        return py_trees.common.Status.RUNNING

        
    def SimulateMovement(self):
        logger.info('SimulateMovement')
        return py_trees.common.Status.SUCCESS
    
def TickOnceTest(tree):
    tree.tick_once()
    logger.info("tick")
    tree.tick_once()
    logger.info("tick")
    tree.tick_once()
    logger.info("tick")
    tree.tick_once()
    logger.info("tick")
    
def TickTest(tree):
    tree.tick()
    logger.info("tick")
    tree.tick()
    logger.info("tick")
    tree.tick()
    logger.info("tick")
    tree.tick()
    logger.info("tick")
    

    

if __name__ == "__main__":
    logging.basicConfig(filename='PyTreesTest.log', level=logging.INFO)
    
    tree = Sequence("Main", children=[
                    Mapping("map the environment"),
                    Navigation("simulate movement")
                ], memory=True)
    
    tree.setup_with_descendants()
    
    
    


# tree = Sequence("Main", children=[
#     Selector("Does map exist?", children=[
#                 DoesMapExist("Test for Map"),
#                 Parallel("Mapping", policy=py_trees.common.ParallelPolicy.SuccessOnOne(), children=[
#                     Mapping("map the environment", blackboard),
#                     Navigation("move around the table", blackboard)
#                 ])
#                 ], memory=True),
#     Planning("compute path to lower left corner", blackboard, (-1.46, -3.12)),
#     Navigation("move to lower left corner", blackboard),
#     Planning("compute path to sink", blackboard, (0.88, 0.09)),
#     Navigation("move to sink", blackboard)
# ], memory=True)