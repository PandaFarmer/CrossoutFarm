
import pynput
from pynput import keyboard
from threading import Thread
import time
from NodeTrees import *
import cv2
import sys
from Utilities import AttrDict
from Logger import Logger

# https://stackoverflow.com/questions/24072790/how-to-detect-key-presses
# https://stackoverflow.com/questions/11918999/key-listeners-in-python
# https://stackoverflow.com/questions/65068775/how-to-stop-or-pause-pyautogui-at-any-moment-that-i-want

# from pynput.keyboard import Controller, Key, Listener
# from pynput.mouse import Button, Controller

logger = Logger("INFO")
bb_ = AttrDict({'main_status':"running"}) #blackboard

def on_press(key):
    logger.info(msg=f"{key} pressed")
    print(f"{key} pressed")
    
    if key == keyboard.Key.ctrl_r:
        logger.info("")
        if bb_.main_status == 'pause':
            bb_.main_status = 'run'
        else:
            bb_.main_status = 'pause'
    if key == keyboard.Key.esc:
        bb_.main_status = 'exit'
        sys.exit()

def main_status(description):   
    return NodeStatus.RUNNING
      

def main_loop(description):
    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    listener.join()
    print(f"main_status: {bb_.main_status}\ndescription:{description}")
    logger.debug(f"main_status: {bb_.main_status}\ndescription:{description}")

    if bb_.main_status == 'pause':
        time.sleep(1)

    if bb_.main_status == 'exit':
        print('MainLoop closing')
        cv2.destroyAllWindows()
        sys.exit()
    return NodeStatus.RUNNING
    
if __name__ == "__main__":
    # tree = Sequence("Main", children = [
    #     Parallel("main_controls", children = [
    #         Selector("update_mapping", children = [
    #             large_map_update(), 
    #             mini_map_update()
    #         ]),
    #         Sequence("update_object_tracking", children = [
    #             update_entity_detection(),
    #             update_obstacle_detection()
    #         ]),
    #         Parallel("update_controls", children = [
    #             update_camera_tracking_info(),
    #             move_camera(),
    #             update_movement(),
    #             handle_shooting(),
    #             handle_prompt()
    #         ])
    #     ])
    # ], memory=True)
    

    tree = Parallel(description="Main", children = [
        main_status("Listen For Keyboard Prompt Status Updates"), 
        main_loop("Iterating through program functions")])
    
    while True:
        tree.tick()
    
    # main_status = 'run'
    
    # Thread(target=main).start()
    # listen_for_interrupts()