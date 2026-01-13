import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle, Patch
import numpy as np
from classes.Lattice import Lattice
import classes.DLAGrowth as DLA
import classes.EDENGrowth as EDEN

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

def get_grain_boundaries_mask(lattice: Lattice) -> np.array:
    """
    Creates a binary mask that identifies voxels (or pixels) at the edge between two or more cristalline domains.

    Args:
        lattice (Lattice): custom lattice object.

    Returns:
        np.array: binary mask.
    """
    ids = lattice.group_id
    occupied = ids > 0
    pad = np.pad(ids, ((1,1), (1,1), (1,1)), mode='constant', constant_values=0)
    center = pad[1:-1, 1:-1, 1:-1]
    
    n_x_plus  = pad[2:, 1:-1, 1:-1]
    n_x_minus = pad[:-2, 1:-1, 1:-1]
    n_y_plus  = pad[1:-1, 2:, 1:-1]
    n_y_minus = pad[1:-1, :-2, 1:-1]
    n_z_plus  = pad[1:-1, 1:-1, 2:]
    n_z_minus = pad[1:-1, 1:-1, :-2]
    
    neighbors_list = [n_x_plus, n_x_minus, n_y_plus, n_y_minus, n_z_plus, n_z_minus]
    is_boundary_neighbor = np.zeros_like(center, dtype=bool)
    
    for neighbor in neighbors_list:
        neighbor_different = (neighbor != 0) & (neighbor != center)
        is_boundary_neighbor |= neighbor_different
        
    return occupied & is_boundary_neighbor

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
        cmap = plt.cm.get_cmap('tab20b') 
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
            
    elif color_mode == "boundaries":
        pass

    else:
        print(f"Crystal not plot: you choose color_mode = {color_mode}. The only accepted are [epoch, id, boundaries].")
        return

    if three_dim:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        voxels = lattice.grid.astype(bool)

        if not np.any(voxels):
            ax.set_title(title)
            plt.show()
            return

        visible_voxels = get_visible_voxels_binary_mask(lattice)
        colors = np.zeros(lattice.shape + (4,), dtype=np.float32)

        if color_mode == "boundaries":
            boundary_mask = get_grain_boundaries_mask(lattice)
            colors[visible_voxels] = (0.8, 0.8, 0.8, 0.3) 
            colors[boundary_mask] = (1.0, 0.0, 0.0, 1.0)

        elif np.any(visible_voxels):
            surface_values = data_grid[visible_voxels]
            unique_surface_vals = np.unique(surface_values)

            for val in unique_surface_vals:
                mask = (visible_voxels) & (data_grid == val)
                if color_mode == "id":
                    colors[mask] = id_to_color[val]
                else:
                    colors[mask] = cmap(norm(val))
        
        if color_mode in ["id", "boundaries"]:
            edge_color = None; line_width = 0.0
        else:
            edge_color = 'k'; line_width = 0.2

        ax.voxels(visible_voxels, facecolors=colors, edgecolor=edge_color, linewidth=line_width)
        ax.set_title(title)
        ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
        
        nx, ny, nz = lattice.shape
        if len(lattice.occupied) > 0:
            occ_arr = np.array(list(lattice.occupied), dtype=int)
            max_y = int(np.max(occ_arr[:, 1])) if len(occ_arr) > 0 else ny
        else: max_y = 0
        ax.set_xlim(0, nx); ax.set_ylim(0, min(ny, max_y + 5)); ax.set_zlim(0, nz)
        ax.view_init(elev=30, azim=45)

    else:
        fig, ax = plt.subplots(figsize=(8, 7))
        
        if color_mode == "epoch":
            data_2d = np.max(lattice.history, axis=2)
            masked_data = np.ma.masked_where(data_2d < 0, data_2d)
            im = ax.imshow(masked_data.T, origin='lower', cmap=cmap, norm=norm, interpolation='nearest')
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label("Occupation epoch")
            
        elif color_mode == "id":
            data_2d = np.max(lattice.group_id, axis=2)
            masked_data = np.ma.masked_where(data_2d == 0, data_2d)
            
            nx, ny = data_2d.shape
            rgba_img = np.zeros((ny, nx, 4))
            for x in range(nx):
                for y in range(ny):
                    val = data_2d[x, y]
                    if val > 0:
                        rgba_img[y, x] = id_to_color.get(val, (0,0,0,1))
            
            ax.imshow(rgba_img, origin='lower', interpolation='nearest')
            
        elif color_mode == "boundaries":
            # TODO
            pass

        ax.set_title(title)
        ax.set_xlabel('x')
        ax.set_ylabel('y')

    if color_mode == "id":
        legend_elements = [Patch(facecolor=c, edgecolor='k', label=f'ID: {int(uid)}') for uid, c in id_to_color.items()]
        ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), title="Parameters")
    
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
