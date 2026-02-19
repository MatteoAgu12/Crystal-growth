import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import Colormap, Normalize
from matplotlib.patches import Patch
from typing import Union, Tuple, Dict

from classes.BaseLattice import BaseLattice
from classes.KineticLattice import KineticLattice
from GUI.gui_common import set_axes_labels, finalize_plot

import logging
logger = logging.getLogger("growthsim")

def get_visible_voxels_binary_mask(lattice: KineticLattice) -> np.ndarray:
    """
    Create a binary mask showing only the occupied cells not completely surrounded by other occupied cells.

    Args:
        lattice (KineticLattice): custom lattice object containing occupation data

    Return:
        (np.ndarray): boolean 3D mask where True indicates a visible voxel
    """
    occ = lattice.grid.astype(bool)
    p = np.pad(occ, ((1, 1), (1, 1), (1, 1)), mode='constant', constant_values=False)

    n1, n2, n3 = p[2:, 1:-1, 1:-1], p[:-2, 1:-1, 1:-1], p[1:-1, 2:, 1:-1]
    n4, n5, n6 = p[1:-1, :-2, 1:-1], p[1:-1, 1:-1, 2:], p[1:-1, 1:-1, :-2]

    neighbors_all_occupied = n1 & n2 & n3 & n4 & n5 & n6
    return occ & (~neighbors_all_occupied)

def get_grain_boundaries_mask(lattice: BaseLattice) -> np.ndarray:
    """
    Create a binary mask identifying voxels at the edge between different crystalline domains.

    Args:
        lattice (BaseLattice): custom lattice object containing group ID assignments

    Return:
        (np.ndarray): boolean 3D mask marking grain boundaries
    """
    ids = lattice.group_id
    occupied = ids > 0
    pad = np.pad(ids, ((1, 1), (1, 1), (1, 1)), mode='constant', constant_values=0)
    center = pad[1:-1, 1:-1, 1:-1]

    n_list = [pad[2:, 1:-1, 1:-1], pad[:-2, 1:-1, 1:-1], pad[1:-1, 2:, 1:-1],
              pad[1:-1, :-2, 1:-1], pad[1:-1, 1:-1, 2:], pad[1:-1, 1:-1, :-2]]
    
    is_boundary_neighbor = np.zeros_like(center, dtype=bool)
    for neighbor in n_list:
        is_boundary_neighbor |= (neighbor != 0) & (neighbor != center)

    return occupied & is_boundary_neighbor

def _build_id_palette(lattice: KineticLattice) -> Tuple[np.ndarray, Colormap, Dict[int, tuple]]:
    """
    Construct the data grid, colormap, and ID-to-color mapping for ID-based voxel coloring.

    Args:
        lattice (KineticLattice): custom lattice object

    Return:
        (Tuple[np.ndarray, Colormap, dict]): tuple containing the ID grid, colormap object, and dictionary mapping IDs to RGBA colors
    """
    data_grid = lattice.group_id
    cmap = colormaps['tab20b']
    unique_vals = np.unique(data_grid[lattice.grid == 1])
    id_to_color = {uid: cmap(i % cmap.N) for i, uid in enumerate(unique_vals)}
    return data_grid, cmap, id_to_color

def _build_epoch_palette(lattice: KineticLattice, N_epochs: int) -> Tuple[np.ndarray, Colormap, Normalize]:
    """
    Construct the data grid, colormap, and normalization for epoch-based voxel coloring.

    Args:
        lattice (KineticLattice): custom lattice object
        N_epochs (int): total number of epochs in the simulation

    Return:
        (Tuple[np.ndarray, Colormap, Normalize]): tuple containing history grid, colormap, and normalization bounds
    """
    data_grid = lattice.history
    cmap = plt.cm.viridis
    norm = Normalize(vmin=0, vmax=N_epochs)
    return data_grid, cmap, norm

def _render_voxels_boundaries(ax: plt.Axes, lattice: KineticLattice, visible_voxels: np.ndarray) -> np.ndarray:
    """
    Render grain boundaries in a 3D lattice visualization.

    Args:
        ax (plt.Axes): matplotlib 3D axis object
        lattice (KineticLattice): custom lattice object
        visible_voxels (np.ndarray): binary mask of visible exterior voxels

    Return:
        (np.ndarray): 4D array of RGBA colors applied to the rendered voxels
    """
    boundary_mask = get_grain_boundaries_mask(lattice)
    plot_mask = visible_voxels | boundary_mask

    colors = np.zeros(lattice.shape + (4,), dtype=np.float64)
    colors[visible_voxels] = (0.9, 0.9, 0.9, 0.1)
    colors[boundary_mask] = (0.0, 0.0, 0.0, 1.0)

    ax.voxels(plot_mask, facecolors=colors, edgecolor=None, linewidth=0)
    return colors

def _render_voxels_surface(ax: plt.Axes, lattice: KineticLattice, visible_voxels: np.ndarray, data_grid: np.ndarray, 
                           color_mode: str, cmap: Colormap, norm_or_id_to_color: Union[Normalize, dict]) -> np.ndarray:
    """
    Render voxel surfaces in a 3D lattice based on a specified continuous or discrete color mapping.

    Args:
        ax (plt.Axes): matplotlib 3D axis object
        lattice (KineticLattice): custom lattice object
        visible_voxels (np.ndarray): binary mask of visible voxels
        data_grid (np.ndarray): 3D grid containing data values for coloring
        color_mode (str): coloring strategy ("id" or "epoch")
        cmap (Colormap): matplotlib colormap object
        norm_or_id_to_color (Normalize or dict): scaling normalizer or dictionary mapping IDs to colors

    Return:
        (np.ndarray): 4D array of RGBA colors mapped to the active voxels
    """
    colors = np.zeros(lattice.shape + (4,), dtype=np.float64)

    if np.any(visible_voxels):
        surface_values = data_grid[visible_voxels]
        unique_surface_vals = np.unique(surface_values)

        for val in unique_surface_vals:
            mask = (visible_voxels) & (data_grid == val)
            if color_mode == "id":
                colors[mask] = norm_or_id_to_color[val]
            else:
                colors[mask] = cmap(norm_or_id_to_color(val))

    return colors

def _plot_2d_epoch(ax: plt.Axes, lattice: KineticLattice, cmap: Colormap, norm: Normalize) -> None:
    """
    Plot a 2D projection of the kinetic lattice colored by occupation epoch.

    Args:
        ax (plt.Axes): matplotlib axes object
        lattice (KineticLattice): custom lattice object
        cmap (Colormap): colormap for epoch representation
        norm (Normalize): normalization bounds for the epoch range
    """
    data_2d = np.max(lattice.history, axis=2)
    masked_data = np.ma.masked_where(data_2d < 0, data_2d)
    im = ax.imshow(masked_data.T, origin='lower', cmap=cmap, norm=norm, interpolation='nearest')
    plt.colorbar(im, ax=ax, shrink=0.8).set_label("Occupation epoch")

def _plot_2d_id(ax: plt.Axes, lattice: KineticLattice, id_to_color: dict) -> None:
    """
    Plot a 2D projection of the kinetic lattice colored by unique grain IDs.

    Args:
        ax (plt.Axes): matplotlib axes object
        lattice (KineticLattice): custom lattice object
        id_to_color (dict): dictionary mapping discrete grain IDs to RGBA tuples
    """
    data_2d = np.max(lattice.group_id, axis=2)
    masked_data = np.ma.masked_where(data_2d == 0, data_2d)

    nx, ny = data_2d.shape
    rgba_img = np.zeros((ny, nx, 4))
    for x in range(nx):
        for y in range(ny):
            val = data_2d[x, y]
            if val > 0:
                rgba_img[y, x] = id_to_color.get(val, (0, 0, 0, 1))

    ax.imshow(rgba_img, origin='lower', interpolation='nearest')

def _plot_2d_boundaries(ax: plt.Axes, lattice: KineticLattice) -> None:
    """
    Plot a 2D projection emphasizing the physical boundaries between grains.

    Args:
        ax (plt.Axes): matplotlib axes object
        lattice (KineticLattice): custom lattice object
    """
    boundary_mask = get_grain_boundaries_mask(lattice)
    mask_2d = np.any(boundary_mask, axis=2)
    occ_2d = np.any(lattice.grid, axis=2)

    nx, ny = mask_2d.shape
    rgba_img = np.ones((ny, nx, 4))

    rgba_img[occ_2d.T] = [0.9, 0.9, 0.9, 1.0]
    rgba_img[mask_2d.T] = [0.0, 0.0, 0.0, 1.0]

    ax.imshow(rgba_img, origin='lower', interpolation='nearest')

def plot_kinetic_lattice(lattice: KineticLattice, N_epochs: int, title: str, out_dir: Union[str, None], 
                         three_dim: bool, color_mode: str = "epoch") -> None:
    """
    Create a 2D or 3D visualization of the discrete kinetic lattice simulation.

    Args:
        lattice (KineticLattice): custom lattice object holding simulation data
        N_epochs (int): total number of epochs run in the simulation
        title (str): descriptive title for the generated plot
        out_dir (str or None): output directory path for saving the image file
        three_dim (bool): flag enabling 3D volumetric plotting
        color_mode (str): rendering mode, accepts 'epoch', 'id', or 'boundaries'
    """
    id_to_color = None
    data_grid = None
    cmap = None
    norm = None

    if color_mode == "id":
        data_grid, cmap, id_to_color = _build_id_palette(lattice)
    elif color_mode == "epoch":
        data_grid, cmap, norm = _build_epoch_palette(lattice, N_epochs)
    elif color_mode == "boundaries":
        pass
    else:
        logger.warning("Crystal not plot: you choose color_mode = %s. The only accepted are [epoch, id, boundaries].", color_mode)
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

        if color_mode == "boundaries":
            colors = _render_voxels_boundaries(ax, lattice, visible_voxels)
        else:
            colors = _render_voxels_surface(
                ax=ax, lattice=lattice, visible_voxels=visible_voxels,
                data_grid=data_grid, color_mode=color_mode, cmap=cmap,
                norm_or_id_to_color=(id_to_color if color_mode == "id" else norm)
            )

        line_width = 0.0 if color_mode in ["id", "boundaries"] else 0.2

        ax.voxels(visible_voxels, facecolors=colors, edgecolor=None, linewidth=line_width)
        ax.set_title(title)
        set_axes_labels(ax, is_3d=True)

        nx, ny, nz = lattice.shape
        if len(lattice.occupied) > 0:
            occ_arr = np.array(list(lattice.occupied), dtype=int)
            max_y = int(np.max(occ_arr[:, 1])) if len(occ_arr) > 0 else ny
        else:
            max_y = 0
            
        ax.set_xlim(0, nx); ax.set_ylim(0, min(ny, max_y + 5)); ax.set_zlim(0, nz)
        ax.view_init(elev=30, azim=45)

    else:
        fig, ax = plt.subplots(figsize=(8, 7))

        if color_mode == "epoch":
            _plot_2d_epoch(ax, lattice, cmap, norm)
        elif color_mode == "id":
            _plot_2d_id(ax, lattice, id_to_color)
        elif color_mode == "boundaries":
            _plot_2d_boundaries(ax, lattice)

        ax.set_title(title)
        set_axes_labels(ax, is_3d=False)

    if color_mode == "id":
        legend_elements = [Patch(facecolor=c, edgecolor='k', label=f'ID: {int(uid)}') for uid, c in id_to_color.items()]
        ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), title="Parameters")

    finalize_plot(out_dir, title, "", "KineticLattice image saved as %s!")