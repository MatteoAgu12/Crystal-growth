import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from Lattice import Lattice

def get_visible_voxels_binary_mask(lattice: Lattice) -> np.array:
    """
    Function that creates a binary mask based on the lattice occupation.
    This function creates a mask that when applied to the lattice, return only the occupied cells not surrounded by other occupied cell.
    The porpouse of this function is to improve the GUI computational time and cost, by avoiding to show voxels that can't be seen.

    Args:
        lattice (Lattice): custom lattice object

    Returns:
        (np.array): binary 3D mask. The cell is True if it is going to be plotted, False elsewhere.
    """
    original_nx, original_ny, original_nz = lattice.shape
    binary_mask = np.zeros_like(lattice.grid, dtype=bool)
    directions = [
        (1, 0, 0), (-1, 0, 0),
        (0, 1, 0), (0, -1, 0),
        (0, 0, 1), (0, 0, -1)
    ]
    occupied = lattice.grid.astype(bool)
    iterator = np.nditer(occupied, flags=['multi_index'])
    
    while not iterator.finished:
        if iterator[0]:
            x, y, z = iterator.multi_index
            for dx, dy, dz in directions:
                nx, ny, nz = x + dx, y + dy, z + dz
                if (0 <= nx < original_nx and 0 <= ny < original_ny and 0 <= nz < original_nz):
                    if not occupied[nx, ny, nz]:
                        binary_mask[x, y, z] = True
                        break
                
                else:
                    binary_mask[x, y, z] = True
                    break
        
        iterator.iternext()
        
    return binary_mask

def plot_lattice(lattice: Lattice, title: str = "Crystal lattice"):
    """
    Function that creates an interactive window that plots the 3D lattice.
    The user can inspect it (zoom, rotate, ...) at runtime.

    Args:
        lattice (Lattice): lattice to be plotted
        title (str, optional): Title of the canvas. Defaults to "Crystal lattice".
    """
    x, y, z = np.nonzero(lattice.grid)
    
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    voxels = lattice.grid.astype(bool)
    facecolors = np.where(voxels, 'royalblue', 'none')
    visible_voxels = get_visible_voxels_binary_mask(lattice)
    ax.voxels(visible_voxels, facecolors=facecolors, edgecolor='k', linewidth=0.2)
    
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
    