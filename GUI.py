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

def plot_lattice(lattice: Lattice, N_epochs: int, title: str = "Crystal lattice", out_dir: str = None, three_dim : bool = True):
    """
    Function that creates an interactive window that plots the 3D (or 2D) lattice.
    The user can inspect it (zoom, rotate -only in 3D version-, ...) at runtime.

    Args:
        lattice (Lattice): lattice to be plotted.
        N_epochs (int): total number of epochs in the simulation.
        title (str, optional): Title of the canvas. Defaults to "Crystal lattice".
        out_dir (str, optional): directory in which save an image of the crystal. Default to None.
        three_dim (bool, optional): decides if the crystal is two or three dimentional. Defaults to True.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize
    from matplotlib.patches import Rectangle

    history = lattice.history

    cmap = plt.cm.viridis
    norm = Normalize(vmin=0, vmax=N_epochs)

    if three_dim:
        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(111, projection='3d')

        # boolean occupancy grid
        voxels = lattice.grid.astype(bool)

        # if nothing occupied -> show empty axes and return
        if not np.any(voxels):
            ax.set_title(title)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            plt.tight_layout()
            if out_dir is not None:
                filename = out_dir + "Crystal.png"
                plt.savefig(filename)
                print(f"Lattice image saved as {filename}!")
            plt.show()
            return

        # -------------------------
        # VECTORIZED: compute visible (surface) mask in 3D
        # A voxel is visible if it is occupied AND at least one 6-neighbor is empty
        occ = voxels
        p = np.pad(occ, ((1,1),(1,1),(1,1)), mode='constant', constant_values=False)
        n_xp = p[2:,1:-1,1:-1]
        n_xm = p[:-2,1:-1,1:-1]
        n_yp = p[1:-1,2:,1:-1]
        n_ym = p[1:-1,:-2,1:-1]
        n_zp = p[1:-1,1:-1,2:]
        n_zm = p[1:-1,1:-1,:-2]
        neighbors_all_occupied = n_xp & n_xm & n_yp & n_ym & n_zp & n_zm
        visible_voxels = occ & (~neighbors_all_occupied)
        # -------------------------

        # Build RGBA color array (float32) instead of dtype=object
        colors = np.zeros(lattice.shape + (4,), dtype=np.float32)  # default transparent

        # unique epoch values among visible occupied voxels
        if np.any(visible_voxels):
            unique_epochs = np.unique(history[visible_voxels])
            # assign color per epoch (batch)
            for epoch in unique_epochs:
                mask = (visible_voxels) & (history == epoch)
                rgba = cmap(norm(epoch))  # returns (r,g,b,a) floats 0..1
                colors[mask] = rgba

        # draw voxels with efficient float RGBA facecolors
        ax.voxels(visible_voxels, facecolors=colors, edgecolor='k', linewidth=0.2)

        ax.set_title(title)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        nx, ny, nz = lattice.shape
        if len(lattice.occupied) > 0:
            try:
                occ_arr = np.array(list(lattice.occupied), dtype=int)
                max_y_occ = int(np.max(occ_arr[:, 1]))
            except Exception:
                max_y_occ = max(t[1] for t in lattice.occupied)
        else:
            max_y_occ = 0

        y_limit = min(ny, max_y_occ + 5)  # +2 rispetto alla cella più alta, senza superare ny

        try:
            ax.set_box_aspect((nx, ny, nz))
        except Exception:
            ax.set_xlim(0, nx)
            ax.set_ylim(0, y_limit)
            ax.set_zlim(0, nz)

        ax.set_xlim(0, nx)
        ax.set_ylim(0, ny)
        ax.set_zlim(0, nz)

        ax.set_xticks([0, max(0, nx // 2), max(0, nx - 1)])
        ax.set_yticks([0, max(0, ny // 2), max(0, ny - 1)])
        ax.set_zticks([0, max(0, nz // 2), max(0, nz - 1)])

        ax.set_xticklabels([str(0), str(nx // 2), str(nx - 1)])
        ax.set_yticklabels([str(0), str(ny // 2), str(ny - 1)])
        ax.set_zticklabels([str(0), str(nz // 2), str(nz - 1)])

        ax.view_init(elev=30, azim=45)

        # Scalarmap
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # matplotlib requires set_array even if empty
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label("Occupation epoch")
        plt.tight_layout()

        if out_dir is not None:
            filename = out_dir + "Crystal.png"
            plt.savefig(filename)
            print(f"Lattice image saved as {filename}!")
        plt.show()

    else:
        # 2D plotting kept mostly as you wrote it, with a small guard if no occupied pixels
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.set_title(title)
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        seeds = lattice.get_nucleation_seeds()
        if seeds is None or len(seeds) == 0:
            # nothing to plot
            ax.set_aspect('equal')
            plt.tight_layout()
            if out_dir is not None:
                filename = out_dir + "Crystal.png"
                plt.savefig(filename)
                print(f"Lattice image saved as {filename}!")
            plt.show()
            return

        z_coord = seeds[0][2]
        slice_z = lattice.grid[:, :, z_coord]
        occupied = np.argwhere(slice_z)

        if occupied.size == 0:
            ax.set_aspect('equal')
            plt.tight_layout()
            if out_dir is not None:
                filename = out_dir + "Crystal.png"
                plt.savefig(filename)
                print(f"Lattice image saved as {filename}!")
            plt.show()
            return

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

        if out_dir is not None:
            filename = out_dir + "Crystal.png"
            plt.savefig(filename)
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
