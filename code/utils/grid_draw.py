import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

nrows, ncols = 8, 10
cmap = ListedColormap(['white', 'black', 'red', 'green', 'blue'])
colors_len = 5

brush_size = 1  # default brush size
brush_color = 0  # default brush color (white)

fig, ax = plt.subplots()

vals = 1
image = np.array(np.random.choice(vals, (nrows, ncols)))


mat = ax.matshow(image, cmap=cmap, vmin=0, vmax=colors_len)
ax.set_xticks(range(ncols))
ax.set_yticks(range(nrows))
ax.set_xticks(np.arange(-.5, ncols, 1), minor=True)
ax.set_yticks(np.arange(-.5, nrows, 1), minor=True)

ax.grid(which='minor', color='k', linestyle='-', linewidth=2)
ax.tick_params(which='both', axis='both', length=0) # don't show tick marks

def update_grid(event):
    # Get the mouse click coordinates
    if event.inaxes == ax:
        x, y = int(event.xdata+0.5), int(event.ydata +0.5)

        for i in range(-brush_size + 1, brush_size):
                for j in range(-brush_size + 1, brush_size):
                    if 0 <= x + i < ncols and 0 <= y + j < nrows:
                        image[y + j, x + i] = brush_color

        mat.set_array(image)
        plt.draw()


def export_grid(image):
    with open('exported_grid.py', 'w') as file:
        file.write('grid = [\n')
        for row in image:
            file.write('    ' + str(list(row)) + ',\n')
        file.write(']\n')

def on_close(event):
    export_grid(image)
    print("exported_grid.py has been created with the current grid values.")
    print(image)

def on_key(event):
    global brush_size
    if event.key in ['1', '2', '3', '4', '5', '6']:
        brush_size = int(event.key)
    
    global brush_color
    if event.key in ['z', 'x', 'c', 'v', 'b', 'n']:
        key_to_color = {'z': 0, 'x': 1, 'c': 2, 'v': 3, 'b': 4, 'n': 5}  # Map keys to color indices
        brush_color = key_to_color[event.key] % colors_len
        
    print(f'Brush size = {brush_size}, Brush Color = {brush_color}')

fig.canvas.mpl_connect('button_press_event', update_grid)
fig.canvas.mpl_connect('close_event', on_close)
fig.canvas.mpl_connect('key_press_event', on_key)

plt.show()

