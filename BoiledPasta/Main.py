
import pynput
from pynput import keyboard
from threading import Thread
import time
import py_trees
from py_trees.composites import Sequence, Selector, Parallel
from py_trees.behaviour import Behaviour
import cv2
from Logger import Logger
import sys
from Utilities import AttrDict

# https://stackoverflow.com/questions/24072790/how-to-detect-key-presses
# https://stackoverflow.com/questions/11918999/key-listeners-in-python
# https://stackoverflow.com/questions/65068775/how-to-stop-or-pause-pyautogui-at-any-moment-that-i-want

# from pynput.keyboard import Controller, Key, Listener
# from pynput.mouse import Button, Controller


bb_ = AttrDict({'main_status':"pause"}) #blackboard
   
class MainStatus(Behaviour):
    def __init__(self, name, bb_):
        super(MainStatus, self).__init__(name)
        self.bb_ = bb_
        self.listener = keyboard.Listener(on_press=self.on_press)
        self.listener.start()
        self.listener.join()
        self.module_name = "MainStatus"
        self.logger = Logger(module_name=self.module_name, shared_logging =True, logging_level="INFO")
        
    def on_press(self, key):
        self.logger.info(msg=f"{key} pressed")
        print('{0} pressed'.format(
            key))
        if key == keyboard.Key.ctrl_r:
            if self.bb_.main_status == 'pause':
                self.bb_.main_status = 'run'
            else:
                self.bb_.main_status = 'pause'
        if key == keyboard.Key.esc:
            self.bb_.main_status = 'exit'
            sys.exit()
        
    def update(self):
        self.logger.info(f"module: {self.module_name} update...{self.name}")
        return py_trees.common.Status.RUNNING
        
    def setup(self):
        self.logger.info(f"module: {self.module_name} setup...{self.name}")
        
    def initialize(self):
        self.logger.info(f"module: {self.module_name} initialize...{self.name}")
        
    def update(self):
        self.logger.info(f"module: {self.module_name} update...{self.name}")
        
    def terminate(self):
        self.logger.info(f"module: {self.module_name} terminate...{self.name}")
        
class MainLoop(Behaviour):
    def __init__(self, name, blackboard):
        super(MainLoop, self).__init__(name)
        self.blackboard = blackboard
        # self.listener = keyboard.Listener(on_press=self.on_press)
        # self.listener.start()
        # self.listener.join()
        self.module_name = "MainLoop"
        self.logger = Logger(module_name=self.module_name, shared_logging =True, logging_level="INFO")
        
    def update(self):
        self.logger.info(self.module_name, f"update...{self.name}")
        print(f"main_status: {bb_.main_status}")

        if bb_.main_status == 'pause':
            time.sleep(1)

        if bb_.main_status == 'exit':
            print('MainLoop closing')
            cv2.destroyAllWindows()
            self.terminate()
            sys.exit()
        return py_trees.common.Status.RUNNING
        
    def setup(self):
        self.logger.info(self.module_name, f"setup...{self.name}")
        
    def initialize(self):
        self.logger.info(self.module_name, f"initialize...{self.name}")
        
    def update(self):
        self.logger.info(self.module_name, f"update...{self.name}")
        
    def terminate(self):
        self.logger.info(self.module_name, f"terminate...{self.name}")
    
if __name__ == "__main__":
    # tree = Sequence("Main", children = [
    #     Parallel("MainControls", children = [
    #         Selector("UpdateMapping", children = [
    #             LargeMapUpdate(), 
    #             MiniMapUpdate()
    #         ]),
    #         Sequence("UpdateObjectTracking", children = [
    #             UpdateEntityDetection(),
    #             UpdateObstacleDetection()
    #         ]),
    #         Parallel("UpdateControls", children = [
    #             UpdateCameraTrackingInfo(),
    #             MoveCamera(),
    #             UpdateMovement(),
    #             HandleShooting(),
    #             HandlePrompt()
    #         ])
    #     ])
    # ], memory=True)
    
    # tree = Parallel("Main", children = [
    #     main
    # ])
    tree = Parallel("Main", policy=py_trees.common.ParallelPolicy.SuccessOnAll(), children = [
        MainStatus("Listen For Keyboard Prompt Status Updates", bb_), 
        MainLoop("Iterating through program functions", bb_)])
    
    while True:
        tree.tick_once()
    
    # main_status = 'run'
    
    # Thread(target=main).start()
    # listen_for_interrupts()