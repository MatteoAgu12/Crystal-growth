import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from Lattice import Lattice

def plot_lattice(lattice: Lattice, title: str = "Crystal lattice"):
    """_summary_

    Args:
        lattice (Lattice): _description_
        title (str, optional): _description_. Defaults to "Crystal lattice".
    """
    x, y, z = np.nonzero(lattice.grid)
    
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c='dodgerblue', marker='o', s=10, alpha=0.8)
    
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    
    max_range = max(lattice.shape)
    ax.set_xlim(0, max_range)
    ax.set_ylim(0, max_range)
    ax.set_zlim(0, max_range)
    
    ax.view_init(elev=30, azim=45)
    plt.tight_layout()
    plt.show()