#issues commands attempting to situate vehicle in an optimal spot for destroying enemies, 
# following teammates, and/or completing objectives, while avoiding obstacles and

import pyautogui

def update_movement(hostile_positions, ally_positions, vehicle_health_percentage, obstacle_positions):
    """update movement based on recommended pathing and positional heuristics"""
    
    
def update_mouse_movement():
    """"""

mouse = pynput.mouse.Controller()
l_mouse_button = pynput.mouse.Button.left
r_mouse_button = pynput.mouse.Button.right

def move_mouse(x, y):
    mouse.move(x, y)

movement_dir = {'w':(0, 1), 'a':(-1, 0), 's':(0, -1), 'd':(1, 0)}
movement = {'w', 's', 'a', 'd'}
weapons_and_gadgets = {'lclick', 'rclick', 'mouse4', '1', '2'}
prompts = {'tab':'\t', 'interact':'r'}
# interrupts = {'pause':'backspace', 'quit':'cmd'}