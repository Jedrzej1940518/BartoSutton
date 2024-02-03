
from dataclasses import dataclass
import random
from typing import List
import exported_grid
import numpy as np

grid = np.array(exported_grid.grid)

@dataclass(frozen=True)  
class State:
    x: int
    y: int

def get_terminal_state() -> State:
    
    for y in range(exported_grid.nrows - 1): ## no windy row
        for x in range(exported_grid.ncols):
            if grid[y, x] == exported_grid.colors['blue']:
                return State(x, y)

    print("error terminal space")
    return State(-1,-1)

def get_starting_state() -> State:
    
    for y in range(exported_grid.nrows - 1): ## no windy row
        for x in range(exported_grid.ncols):
            if grid[y, x] == exported_grid.colors['green']:
                return State(x, y)

    print("error starting space")
    return State(-1,-1)

def get_state_space() -> List[State]:
    
    space = []
    
    for y in range(exported_grid.nrows - 1): ## no windy row
        for x in range(exported_grid.ncols):
            space.append(State(x, y))

    return space

state_space = get_state_space()
terminal_state = get_terminal_state()
starting_state = get_starting_state()


def get_action_space(state: State):
    x = state.x
    y = state.y
    actions = []
    
    if x > 0:
        actions.append(3)
    if x < exported_grid.ncols -1:
        actions.append(5)
    if y > 0:
        actions.append(1)
    if y < exported_grid.wind_row -1:
        actions.append(7)
        
    if x > 0 and y > 0:
        actions.append(0)
    if x < exported_grid.ncols -1 and  y > 0:
        actions.append(2)
    if x < exported_grid.ncols -1 and y < exported_grid.wind_row -1:
        actions.append(8)
    if x>0 and y < exported_grid.wind_row -1:
        actions.append(6)
        
    actions.append(4)

    return actions

def get_wind(state: State) -> int:
    c = random.random()
    w = grid[exported_grid.wind_row, state.x]
    if c < 0.33:
        w = 0
    elif c < 0.66:
        w = w
    else:
        w= w*2
        
    return w

def get_r_s_a(state: State, action, policy) -> [int, State, int]:
    
    wind = get_wind(state)

    x, y = state.x, state.y
    
    if action == 1:
        y -= 1
    elif action == 7:
        y += 1
    elif action == 3:
        x -= 1
    elif action == 5:
        x += 1
        
    elif action == 0:
        y -= 1
        x -= 1
    elif action == 2:
        x += 1
        y -= 1
    elif action == 6:
        y += 1
        x -= 1
    elif action == 8:
        x += 1
        y += 1


    y-= wind
    r = -1
    
    x = max(min(exported_grid.ncols -1, x), 0)
    y = max(min(exported_grid.wind_row -1, y), 0)
    
    state = State(x,y)
    if state == terminal_state:
      r = 1  
    
    a = policy(state)
    
    return [r, state, a]

def e_greedy(e, Q_s_a, state):

    action_space = get_action_space(state)
    if random.random() < e:
        return random.choice(action_space)
    
    max_a = random.choice(action_space)
    max_val = Q_s_a[state,max_a]
    
    for a in action_space:
        val = Q_s_a[state,a]
        if val > max_val:
            max_a = a
            max_val = val
                
    if max_a == -1:
        print("error a = -1")    
        
    return max_a

#on-policy TD control
#a = step_size
def sarsa(episode_number, alpha, gamma):
    
    Q_s_a = {}

    for s in state_space:
        for a in get_action_space(s):
           Q_s_a[s,a] = 0  #to do try different
    
    for a in get_action_space(terminal_state):      
        Q_s_a[s,a] = 0
    
    print('SARSA')
    
    for i in range(episode_number):
        
        e = 0.1 #1 - i / episode_number
        policy = lambda s : e_greedy(e, Q_s_a, s)
        
        s = starting_state
        a = policy(s)
        
        n = 0
        
        while s != terminal_state:
            [r, s_n, a_n ] = get_r_s_a(s, a, policy)
            Q_s_a[s,a] = Q_s_a[s,a] + alpha*(r + gamma * Q_s_a[s_n, a_n] - Q_s_a[s, a])
            s = s_n
            a = a_n
            n +=1
            if n > 100000:
                print("yo error!")
    
        print(f'{i}={n}')
    
    

    pass

def main():
    sarsa(200, 0.5, 1)
    sarsa(200, 0.5, 1)
    
main()