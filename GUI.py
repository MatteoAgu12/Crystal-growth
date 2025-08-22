import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle
import numpy as np
from Lattice import Lattice
import DLA_simulation as DLA
import EDEN_simulation as EDEN

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

def plot_lattice(lattice: Lattice, N_epochs: int, title: str = "Crystal lattice", three_dim : bool = True):
    """
    Function that creates an interactive window that plots the 3D (or 2D) lattice.
    The user can inspect it (zoom, rotate -only in 3D version-, ...) at runtime.

    Args:
        lattice (Lattice): lattice to be plotted.
        N_epochs (int): total number of epochs in the simulation.
        title (str, optional): Title of the canvas. Defaults to "Crystal lattice".
        three_dim (bool, optional): decides if the crystal is two or three dimentional. Defaults to True.
    """
    history = lattice.history
    
    cmap = plt.cm.viridis
    norm = Normalize(vmin=0, vmax=N_epochs)
    
    if three_dim:    
        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(111, projection='3d')
    
        voxels = lattice.grid.astype(bool)
        visible_voxels_indices = np.where(voxels)
        colors = np.empty(lattice.shape, dtype=object)
        
        for i in range(len(visible_voxels_indices[0])):
            epoch = history[visible_voxels_indices[0][i], visible_voxels_indices[1][i], visible_voxels_indices[2][i]]
            color = cmap(norm(epoch))
            colors[visible_voxels_indices[0][i], visible_voxels_indices[1][i], visible_voxels_indices[2][i]] = color
        
        visible_voxels = get_visible_voxels_binary_mask(lattice)
        ax.voxels(visible_voxels, facecolors=colors, edgecolor='k', linewidth=0.2)
    
        ax.set_title(title)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
    
        max_range = max(lattice.shape)
        ax.set_xlim(0, max_range)
        ax.set_ylim(0, max_range)
        ax.set_zlim(0, max_range)    
        ax.view_init(elev=30, azim=45)
        
        # Scalarmap
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label("Occupation epoch")
        
        plt.tight_layout()
        plt.show()
        
    else:
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.set_title(title)
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        z_coord = lattice.get_nucleation_seeds()[0][2]
        slice_z = lattice.grid[:, :, z_coord]
        occupied = np.argwhere(slice_z)
        x_list = occupied[:, 0].tolist()
        y_list = occupied[:, 1].tolist()

        for x, y in zip(x_list, y_list):
            epoch = history[x, y, z_coord]
            color = cmap(norm(epoch))
            ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor=color, edgecolor='black'))

        ax.set_xlim(min(x_list) - 1, max(x_list) + 1)
        ax.set_ylim(min(y_list) - 1, max(y_list) + 1)
        ax.set_aspect('equal')
        ax.grid(False)
        
        # Scalarmap
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label("Occupation epoch")

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':    
    """LATTICE = Lattice(5, 2, 2)
    LATTICE.occupy(2, 0, 0, 2)
    LATTICE.occupy(2, 0, 1, 3)
    plot_lattice(LATTICE)"""
    
    # DLA
    """LATTICE = Lattice(100, 100, 100)
    LATTICE.set_nucleation_seed(50, 50, 50)
    N_EPOCHS = 300
    s_mean, s_std, r_mean, r_std = DLA.DLA_simulation(LATTICE, N_EPOCHS, 1, 3, three_dim=True)
    plot_lattice(LATTICE, N_EPOCHS)"""
    
    # EDEN
    """LATTICE = Lattice(100, 100, 100)
    LATTICE.set_nucleation_seed(50, 50, 50)
    N_EPOCHS = 300
    _ = EDEN.EDEN_simulation(LATTICE, N_EPOCHS, False)
    plot_lattice(LATTICE, N_EPOCHS, three_dim=False)"""