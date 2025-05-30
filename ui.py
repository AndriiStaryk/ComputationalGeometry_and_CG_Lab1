import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np

def plot_ellipse(B, d):
    from get_axes import get_axes

    axes = get_axes(B, d)
    angle = np.degrees(np.arctan2(B[1, 0], B[0, 0]))

    fig, ax = plt.subplots()
    ellipse = Ellipse(xy=d, width=2*axes[0], height=2*axes[1], angle=angle,
                      edgecolor='r', fc='None', lw=2)
    ax.add_patch(ellipse)
    ax.set_xlim(d[0] - axes[0]*2, d[0] + axes[0]*2)
    ax.set_ylim(d[1] - axes[1]*2, d[1] + axes[1]*2)
    ax.set_aspect('equal')
    plt.grid(True)
    plt.title("Maximum Volume Ellipse")
    plt.show()