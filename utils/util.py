import numpy as np
import matplotlib.pyplot as plt

#MT: added
def plot_spectro(x_m, fs, extlmax=None, title='title', vmin=None, vmax=None, save=False):
    if vmin==None:
        vmin = np.min(x_m)
    if vmax==None:
        vmax = np.max(x_m)
    exthmin = 1
    exthmax = len(x_m)
    extlmin = 0
    if extlmax==None:
        extlmax = len(x_m[0])

    plt.figure(figsize=(8, 5))
    plt.imshow(x_m, extent=[extlmin,extlmax,exthmin,exthmax], cmap='inferno',
               vmin=vmin, vmax=vmax, origin='lower', aspect='auto')
    plt.colorbar()
    plt.title(title)
    if save:
        plt.savefig(f"./figures/{title}.png")
    plt.show()
    