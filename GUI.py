import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle, Patch
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
    occ = lattice.grid.astype(bool)
    p = np.pad(occ, ((1,1),(1,1),(1,1)), mode='constant', constant_values=False)
    n1 = p[2:,1:-1,1:-1]
    n2 = p[:-2,1:-1,1:-1]
    n3 = p[1:-1,2:,1:-1]
    n4 = p[1:-1,:-2,1:-1]
    n5 = p[1:-1,1:-1,2:]
    n6 = p[1:-1,1:-1,:-2]
    neighbors_all_occupied = n1 & n2 & n3 & n4 & n5 & n6
    visible = occ & (~neighbors_all_occupied)
    return visible

def plot_lattice(lattice: Lattice, N_epochs: int, title: str = "Crystal lattice", 
                 out_dir: str = None, three_dim : bool = True,
                 color_mode: str = "epoch"):
    """
    Function that creates an interactive window that plots the 3D (or 2D) lattice.
    The user can inspect it (zoom, rotate -only in 3D version-, ...) at runtime.

    Args:
        lattice (Lattice): lattice to be plotted.
        N_epochs (int): total number of epochs in the simulation (used for normalization in 'epoch' mode).
        title (str, optional): Title of the canvas. Defaults to "Crystal lattice".
        out_dir (str, optional): directory in which save an image of the crystal. Default to None.
        three_dim (bool, optional): decides if the crystal is two or three dimentional. Defaults to True.
        color_mode (str, optional): "epoch" to color by history (gradient), "id" to color by parameters (discrete). Defaults to "epoch".
    """
    if color_mode == "id":
        data_grid = lattice.group_id
        cmap = plt.cm.get_cmap('tab20')
        unique_vals = np.unique(data_grid[lattice.grid == 1])
        id_to_color = {uid: cmap(i % cmap.N) for i, uid in enumerate(unique_vals)}
        
        def get_color(val):
            return id_to_color.get(val, (0,0,0,0))
            
    elif color_mode == "epoch":
        data_grid = lattice.history
        cmap = plt.cm.viridis
        norm = Normalize(vmin=0, vmax=N_epochs)
        
        def get_color(val):
            return cmap(norm(val))
        
    else:
        print(f"Crystal not plot: you choose color_mode = {color_mode}. The only accepted are [epoch, id].")
        return

    if three_dim:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        voxels = lattice.grid.astype(bool)

        if not np.any(voxels):
            ax.set_title(title)
            ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
            plt.tight_layout()
            if out_dir: plt.savefig(out_dir + "Crystal.png")
            plt.show()
            return

        occ = voxels
        p = np.pad(occ, ((1,1),(1,1),(1,1)), mode='constant', constant_values=False)
        neighbors_occupied = (p[2:,1:-1,1:-1] & p[:-2,1:-1,1:-1] & 
                              p[1:-1,2:,1:-1] & p[1:-1,:-2,1:-1] & 
                              p[1:-1,1:-1,2:] & p[1:-1,1:-1,:-2])
        visible_voxels = occ & (~neighbors_occupied)
        
        colors = np.zeros(lattice.shape + (4,), dtype=np.float32)

        if np.any(visible_voxels):
            surface_values = data_grid[visible_voxels]
            unique_surface_vals = np.unique(surface_values)

            for val in unique_surface_vals:
                mask = (visible_voxels) & (data_grid == val)
                if color_mode == "id":
                    colors[mask] = id_to_color[val]
                else:
                    colors[mask] = cmap(norm(val))

        ax.voxels(visible_voxels, facecolors=colors, edgecolor='k', linewidth=0.2)

        ax.set_title(title)
        ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
        nx, ny, nz = lattice.shape
        
        if len(lattice.occupied) > 0:
            try:
                occ_arr = np.array(list(lattice.occupied), dtype=int)
                max_y = int(np.max(occ_arr[:, 1]))
            except: max_y = ny
        else: max_y = 0
        
        ax.set_xlim(0, nx)
        ax.set_ylim(0, min(ny, max_y + 5))
        ax.set_zlim(0, nz)
        
        for axis, n in zip([ax.xaxis, ax.yaxis, ax.zaxis], [nx, ny, nz]):
            axis.set_ticks([0, n//2, n-1])
        
        ax.view_init(elev=30, azim=45)

    else:
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.set_title(title)
        ax.set_xlabel('x'); ax.set_ylabel('y')

        seeds = lattice.get_nucleation_seeds()
        
        if seeds is None or len(seeds) == 0 or not np.any(lattice.grid):
            ax.set_aspect('equal')
            if out_dir: plt.savefig(out_dir + "Crystal.png")
            plt.show()
            return

        z_coord = seeds[0][2]
        slice_occ = lattice.grid[:, :, z_coord]
        slice_data = data_grid[:, :, z_coord]
        
        occupied = np.argwhere(slice_occ)

        if occupied.size == 0:
            ax.set_aspect('equal')
            if out_dir: plt.savefig(out_dir + "Crystal.png")
            plt.show()
            return

        for x, y in occupied:
            val = slice_data[x, y]
            if color_mode == "id":
                c = id_to_color.get(val, (0,0,0,1))
            else:
                c = cmap(norm(val))
            
            ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor=c, edgecolor='black'))

        xs, ys = occupied[:, 0], occupied[:, 1]
        ax.set_xlim(xs.min() - 1, xs.max() + 1)
        ax.set_ylim(ys.min() - 1, ys.max() + 1)
        ax.set_aspect('equal')
        ax.grid(False)

    if color_mode == "id":
        legend_elements = [Patch(facecolor=c, edgecolor='k', label=f'ID: {int(uid)}') for uid, c in id_to_color.items()]
        ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), title="Parameters")
    else:
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
        cbar.set_label("Occupation epoch")

    plt.tight_layout()

    if out_dir is not None:
        filename = out_dir + title.replace(" ", "_") + ".png"
        plt.savefig(filename, bbox_inches='tight')
        print(f"Lattice image saved as {filename}!")
    
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
    
    # BATTERY
    """N = 20
    LATTICE = Lattice(N, N, N)
    for x in range(N):
        for z in range(N):
            LATTICE.set_nucleation_seed(x, 1, z)
    N_EPOCHS = 1000
    s_mean, s_std, r_mean, r_std = DLA.DLA_simulation(LATTICE, N_EPOCHS, 1, 3, three_dim=True)
    plot_lattice(LATTICE, N_EPOCHS)"""
