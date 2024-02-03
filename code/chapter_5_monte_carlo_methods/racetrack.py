from dataclasses import dataclass
import random
import exported_grid
import numpy as np

from grid_show import show_grid

from off_policy_monte_carlo_control import OffPolicyMcControl

grid = np.array(exported_grid.grid)
nrows, ncols = 30, 30

colors = {'white' : 0, 'black' : 1, 'red' : 2, 'green': 3, 'blue':4 }

@dataclass(frozen=True)  # Make the dataclass immutable
class State:
    x: int
    y: int
    v_l: int #velocity left
    v_r: int #velocity right
    v_f: int
    win: bool

@dataclass(frozen=True)  # Make the dataclass immutable
class Action:
    v_l_d: int #velocity left delta
    v_r_d: int 
    v_f_d: int

def get_starting_state(win = False) -> State:
    [y,x] = get_starting_position()
    
    return State(x,y, 0,0,0, win)

def get_starting_position() -> [int, int]:
    starting_pos = []
    for y, rows in enumerate(grid):
        for x, columns in enumerate(rows):
            if grid[y,x] == colors['red']:
                starting_pos.append([y,x])
            
    
    return random.choice(starting_pos)
 
def next_state(state: State, action:Action) -> State:

    v_l = state.v_l
    v_r = state.v_r
    v_f = state.v_f
        
    if random.random() > 0.1:
        v_l += action.v_l_d
        v_r += action.v_r_d
        v_f += action.v_f_d

    f = v_f
    r = v_r - v_l
    x = state.x
    y = state.y
    
    death = False
    win = False
    
    while f > 0 or abs(r) > 0:
        if f > 0:
            f-=1
            y -= 1 #minus y goes "up"
        if r > 0:
            x += 1
            r-=1
        elif r < 0:
            x -=1
            r+=1
        
        if x > ncols or x < 0 or y > nrows or y < 0:
            death = True
            break
        
        if grid[y, x] == colors['black']:
            death = True
            break
        
        if grid[y, x] == colors['green']:
            win = True
            break
            
    
    if death:
        return get_starting_state(win)

    return State(x,y,v_l, v_r, v_f, win)

def action_legal(delta, state):
    return state+delta <= 5 and state + delta >=0

def get_action_space(state:State):
   
    actions = []
    for l_d in range(-1, 2):
        for r_d in range(-1, 2):
            for f_d in range(-1,2): #now starting optimal policy will be very very dumb => not move forward
                
                if not action_legal(l_d, state.v_l):
                    continue
                if not action_legal(f_d, state.v_f):
                    continue
                if not action_legal(r_d, state.v_r):
                    continue
                
                actions.append(Action(l_d, r_d, f_d))
    
    return actions

def get_state_space():
    
    print("geting state space")
    
    possible_states = []
    possible_tiles = []
    
    for y in range(nrows):
        for x in range(ncols):
            if grid[y,x] == colors['black']:
                continue
            possible_tiles.append([y,x])
             
    for [y,x] in possible_tiles:
        for v_l in range(0, 6):
            for v_r in range(0,6):
                for v_f in range(0,6):
                    win = grid[y,x] == colors['green']
                    possible_states.append(State(x,y,v_l, v_r, v_f, win))
    
    print("finished getting state space")
    return possible_states

def generate_episode(policy, for_real = False):

    r_s_a = []
    
    s = get_starting_state()
    print(f'Generating episode - starting state {[s.x, s.y]}')

    while True:
        a = policy(s)
        if s.win == True:
            break
        r = -1
        r_s_a.append([r, s, a])
    
        s = next_state(s, a)
        
        if for_real and len(r_s_a) > 30:
            break
    
    r_s_a.append([1, s, a]) #we won
    
    print(f'Episode finished - steps {len(r_s_a)}')
    
    return r_s_a

def main():

    global grid
    
    show_grid(grid)


    gamma = 0.95
    episode_number = 1000
    
    ofpmc = OffPolicyMcControl(gamma, episode_number, get_action_space, get_state_space, generate_episode)
    policy = ofpmc.off_policy_mc()
    callable_policy = lambda s : policy[s]
    print("\nFinished generating policy. Trying optimal policy 4 times: ")

    for i in range(4):
        episode_optimal_policy = generate_episode(callable_policy, True)
        
        print(episode_optimal_policy)
        
        for [r, s, a] in episode_optimal_policy:
            grid[s.y, s.x] = colors['blue']
        
        show_grid(grid)

        grid = np.array(exported_grid.grid)

main()