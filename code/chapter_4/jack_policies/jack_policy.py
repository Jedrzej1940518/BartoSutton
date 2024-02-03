import matplotlib.pyplot as plt
import numpy as np


N = 21

def combine_state(s1:int,s2:int)->int:
    return s1 * N + s2

#maps s to two states s1, s2  
def separate_state(s:int):
    return [s // N, s % N]

def plot_policy_grid(policy, title):
    """
    Plots a 21x21 grid representing the policy. 
    Each cell color corresponds to an action for a state in the policy.

    :param policy: A dictionary mapping from states (0-440) to actions (-5 to 5).
    """
    grid_size = N  # since states range from 0 to 20 in both dimensions
    policy_grid = np.zeros((grid_size, grid_size, 3))  # Initialize a 21x21 grid with RGB color values

    for state, action in policy.items():
        [x, y] = separate_state(state)

        # Map the action to a color (this is a simple mapping, you might want to refine it)
        # Red for -5, Green for 5, intermediate values can be different shades
        red = action / -5
        green = action / 5
        color = [max(red, 0), max(green, 0), 0]
        if color == [0,0,0]:
            color = [1,1,1]
        
        policy_grid[x, y] = color

    # Plotting the grid
    plt.imshow(policy_grid, interpolation='nearest')
    plt.gca().invert_yaxis()
    plt.title(title)

def save_policy(policy, title):
    
    with open(f'output_file_{title}.txt', 'w') as f:  # Open a file in write mode
        plot_policy_grid(policy, title)
        plt.savefig(f'policy_plot_{title}.png')  # Save the plot as an image
        plt.close()  # Close the plot to free up memory

        print(f'-------POLICY {title} -----------\n{policy}\n-------------END-----------------', file=f)
    