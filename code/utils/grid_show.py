import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

def show_grid(grid):
    nrows, ncols = 30, 30
    cmap = ListedColormap(['white', 'black', 'red', 'green', 'blue'])
    colors_len = 5
    image = grid

    fig, ax = plt.subplots()
    mat = ax.matshow(image, cmap=cmap, vmin=0, vmax=colors_len)
    ax.set_xticks(range(ncols))
    ax.set_yticks(range(nrows))
    ax.set_xticks(np.arange(-.5, ncols, 1), minor=True)
    ax.set_yticks(np.arange(-.5, nrows, 1), minor=True)

    ax.grid(which='minor', color='k', linestyle='-', linewidth=2)
    ax.tick_params(which='both', axis='both', length=0) # don't show tick marks


    plt.show()
