import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import colormaps
from matplotlib.colors import Colormap
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import Patch

from skimage import measure
import numpy as np
from scipy.ndimage import map_coordinates

from classes.BaseLattice import BaseLattice
from classes.KineticLattice import KineticLattice
from classes.PhaseFieldLattice import PhaseFieldLattice

# -----------------------------
# Masks / lattice helpers
# -----------------------------
def get_visible_voxels_binary_mask(lattice: KineticLattice) -> np.array:
    """
    Function that creates a binary mask based on the lattice occupation.
    This function creates a mask that when applied to the lattice shows only the occupied cells not
    surrounded by other occupied cell.

    The purpose of this function is to improve the GUI computational time and cost, by avoiding
    to show voxels that can't be seen.

    Args:
        lattice (KineticLattice): custom lattice object

    Returns:
        np.array: binary 3D mask. The cell is True if it is going to be plotted, False elsewhere.
    """
    occ = lattice.grid.astype(bool)
    p = np.pad(occ, ((1, 1), (1, 1), (1, 1)), mode='constant', constant_values=False)

    n1 = p[2:, 1:-1, 1:-1]
    n2 = p[:-2, 1:-1, 1:-1]
    n3 = p[1:-1, 2:, 1:-1]
    n4 = p[1:-1, :-2, 1:-1]
    n5 = p[1:-1, 1:-1, 2:]
    n6 = p[1:-1, 1:-1, :-2]

    neighbors_all_occupied = n1 & n2 & n3 & n4 & n5 & n6
    visible = occ & (~neighbors_all_occupied)
    return visible

def get_grain_boundaries_mask(lattice: BaseLattice) -> np.array:
    """
    Creates a binary mask that identifies voxels (or pixels) at the edge between two or more cristalline domains.

    Args:
        lattice (BaseLattice): custom lattice object.

    Returns:
        np.array: binary mask.
    """
    ids = lattice.group_id
    occupied = ids > 0

    pad = np.pad(ids, ((1, 1), (1, 1), (1, 1)), mode='constant', constant_values=0)
    center = pad[1:-1, 1:-1, 1:-1]

    n_x_plus = pad[2:, 1:-1, 1:-1]
    n_x_minus = pad[:-2, 1:-1, 1:-1]
    n_y_plus = pad[1:-1, 2:, 1:-1]
    n_y_minus = pad[1:-1, :-2, 1:-1]
    n_z_plus = pad[1:-1, 1:-1, 2:]
    n_z_minus = pad[1:-1, 1:-1, :-2]

    neighbors_list = [n_x_plus, n_x_minus, n_y_plus, n_y_minus, n_z_plus, n_z_minus]
    is_boundary_neighbor = np.zeros_like(center, dtype=bool)

    for neighbor in neighbors_list:
        neighbor_different = (neighbor != 0) & (neighbor != center)
        is_boundary_neighbor |= neighbor_different

    return occupied & is_boundary_neighbor

def _mid_plane_z(lattice: BaseLattice) -> int:
    """
    Return the mid-plane index along z, preserving the original logic.
    
    Args:
        lattice (BaseLattice): custom lattice object.

    Returns:
        np.array: coordinate of the z hieght to inspect.
    """
    return 0 if lattice.shape[2] == 1 else lattice.shape[2] // 2


# -----------------------------
# Field / normalization helpers
# -----------------------------
def compute_curvature_2d(phi_slice: np.array) -> np.array:
    """
    Compute the mean curvature H of a 2D slice.
    H = div(grad(phi) / |grad(phi)|)

    Args:
        phi_slice (np.array): custom lattice object.

    Returns:
        np.array: curvature.
    """
    gy, gx = np.gradient(phi_slice)

    norm = np.sqrt(gx ** 2 + gy ** 2) + 1e-8
    nx, ny = gx / norm, gy / norm

    div_nx = np.gradient(nx, axis=1)
    div_ny = np.gradient(ny, axis=0)

    return -(div_nx + div_ny)

def get_field_3d(lattice: PhaseFieldLattice, field_name: str) -> np.array:
    """
    Function that determines which continuous field to analyse.

    Args:
        lattice (PhaseFieldLattice): custom lattice object.
        field_name (str): name of the field

    Returns:
        np.array: array of the requested field.
    """
    if not hasattr(lattice, field_name):
        raise ValueError(f"[GUI] lattice has no field '{field_name}'")
    return getattr(lattice, field_name)

def _get_data_2d_by_name(lattice: PhaseFieldLattice, field_name: str, mid_z: int) -> np.ndarray | None:
    """
    Function that retrieves the 2D data slice from the lattice based on the field name.

    Args:
        lattice (PhaseFieldLattice): custom lattice object.
        field_name (str): name of the field.
        mid_z (int): z index of the mid-plane.

    Returns:
        np.array | None: 2D array of the requested field, or None if unknown
    """
    if field_name == 'phi':
        return lattice.phi[:, :, mid_z]
    if field_name == 'u':
        return lattice.u[:, :, mid_z]
    if field_name == 'curvature':
        return lattice.curvature[:, :, mid_z]
    if field_name == 'history':
        return lattice.history[:, :, mid_z]
    print(f"[GUI] Error: Unknown field {field_name}")
    return None

def _cmap_name_for_mode(color_mode: str | None) -> str:
    """
    Returns the colormap name for the given color mode.
    """
    if color_mode is not None:
        if color_mode == 'phi':
            return 'gray_r'
        if color_mode == 'u':
            return 'inferno'
        if color_mode == 'curvature':
            return 'coolwarm'
        if color_mode == 'history':
            return 'turbo'
        return 'viridis'
    return 'viridis'

def _norm_for_3d_mode(lattice: PhaseFieldLattice, color_mode: str, cvals: np.ndarray) -> Normalize:
    """
    Returns the Normalize object for the given color mode in 3D plotting.

    Args:
        lattice (PhaseFieldLattice): custom lattice object.
        color_mode (str): color mode.
        cvals (np.ndarray): color values.

    Returns:
        Normalize: normalization object.
    """
    if color_mode == 'history':
        h = lattice.history
        global_valid = h[h >= 0]
        if global_valid.size > 0:
            vmin = float(global_valid.min())
            vmax = float(global_valid.max())
            if abs(vmax - vmin) < 1e-12:
                vmax = vmin + 1.0
        else:
            vmin, vmax = 0.0, 1.0
    elif color_mode == 'phi':
        vmin, vmax = 0.0, 1.0
    elif color_mode == 'curvature':
        a = np.abs(cvals)
        vmax = float(np.percentile(a, 95)) if a.size > 0 else 1.0
        vmin = -vmax
    else:
        vmin, vmax = None, None

    return Normalize(vmin=vmin, vmax=vmax)

def _label_for_color_mode_3d(color_mode: str) -> str:
    """
    Returns the colorbar label for the given color mode in 3D plotting.

    Args:
        color_mode (str): color mode.

    Returns:
        str: label string.
    """
    if color_mode == 'phi':
        return "Phase Field $\\phi$"
    if color_mode == 'u':
        return "Field $u$"
    if color_mode == 'curvature':
        return "Curvature / Laplacian"
    if color_mode == 'history':
        return "Solidification Epoch"
    return str(color_mode)


# -----------------------------
# Phase Field plotting
# -----------------------------
def _default_iso_level(vol: np.ndarray) -> float:
    """
    Determine a default iso-level for the marching cubes algorithm based on the volume data.

    Args:
        vol (np.ndarray): 3D volume data.  

    Returns:
        float: default iso-level.
    """
    vmin = float(np.nanmin(vol))
    vmax = float(np.nanmax(vol))
    return vmin + 0.5 * (vmax - vmin) if vmax < 0.5 else 0.5

def plot_2d_phase_field_simulation(lattice: PhaseFieldLattice, out_dir: str,
                                   field_name: str, color_mode: str, title: str):
    """
    Function that creates a 2D plot of the phase field simulation.

    Args:
        lattice (PhaseFieldLattice): custom lattice object.
        out_dir (str): output directory to save the plot. If None, the plot is not saved.
        field_name (str): name of the field to plot.
        color_mode (str): color mode ('phi', 'u', 'curvature', 'history').
        title (str): title of the plot.
    """
    mid_z = _mid_plane_z(lattice)
    data_2d = _get_data_2d_by_name(lattice, field_name, mid_z)
    if data_2d is None:
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    if color_mode == 'phi':
        im = ax.imshow(data_2d.T, origin='lower', cmap='gray_r', vmin=0, vmax=1)
        label = r"Phase Field $\phi$"

    elif color_mode == 'u':
        vmin = -1.0 
        vmax = 1.0  
        im = ax.imshow(data_2d.T, origin='lower', cmap='inferno', vmin=vmin, vmax=vmax)
        label = r"Diffused field $u$"

    elif color_mode == 'curvature':
        curv = compute_curvature_2d(data_2d)
        mask = (data_2d > 0.1) & (data_2d < 0.9)
        curv_masked = np.full_like(curv, np.nan)
        curv_masked[mask] = curv[mask]

        valid_vals = curv[mask]
        if len(valid_vals) > 0:
            vmax = np.percentile(np.abs(valid_vals), 95)
            vmin = -vmax
        else:
            vmin, vmax = -1, 1

        im = ax.imshow(curv_masked.T, origin='lower', cmap='coolwarm', vmin=vmin, vmax=vmax)
        label = r"Curvature $\kappa$"

    elif color_mode == 'history':
        hist = lattice.history[:, :, mid_z].astype(float)
        hist_masked = np.ma.masked_where(hist < 0, hist)

        im = ax.imshow(hist_masked.T, origin='lower', cmap='turbo')
        label = "Solidification Epoch"

    plt.colorbar(im, ax=ax, label=label)
    ax.set_title(f"{title} - {color_mode.capitalize()}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    # ax.contour(data_2d.T, levels=[0.5], colors='white', linewidths=1, origin='lower')

    if out_dir is not None:
        filename = out_dir + title.replace(" ", "_") + '_' + color_mode + ".png"
        plt.savefig(filename, bbox_inches='tight')
        print(f"Phase Field image saved as {filename}!")
    
    plt.show()

def plot_3d_phase_field_simulation(lattice: PhaseFieldLattice, out_dir: str,
                                   field_name: str, color_mode: str, title: str,
                                   iso_level: float = None):
    """
    Function that creates a 3D plot of the phase field simulation using marching cubes.

    Args:
        lattice (PhaseFieldLattice): custom lattice object.
        out_dir (str): output directory to save the plot. If None, the plot is not saved.
        field_name (str): name of the field to plot.
        color_mode (str): color mode ('phi', 'u', 'curvature', 'history').
        title (str): title of the plot.
        iso_level (float, optional): iso-level for the marching cubes algorithm. Defaults to None
    """
    phi = lattice.phi
    nx, ny, nz = phi.shape
    stride = 1

    phi_ds = phi[::stride, ::stride, ::stride]
    vol = phi_ds.astype(np.float32, copy=False)

    if iso_level is None:
        iso_level = _default_iso_level(vol)

    try:
        verts, faces, normals, values = measure.marching_cubes(
            vol, level=float(iso_level), spacing=(stride, stride, stride)
        )
    except Exception as e:
        print(f"[GUI] marching_cubes failed: {e}")
        return

    if color_mode in ('phi', 'u', 'curvature', 'history'):
        cfield = get_field_3d(lattice, color_mode)[::stride, ::stride, ::stride]
    else:
        print(f"[GUI] Unknown color_mode='{color_mode}', Retrn...")
        return

    coords = np.vstack([verts[:, 0], verts[:, 1], verts[:, 2]])
    cvals = map_coordinates(
        cfield.astype(np.float32, copy=False),
        coords,
        order=1,
        mode='nearest'
    )

    if color_mode == 'history':
        cvals = np.where(cvals >= 0, cvals, np.nan)

    cmap_name = _cmap_name_for_mode(color_mode)
    cmap = cm.get_cmap(cmap_name)

    norm = _norm_for_3d_mode(lattice, color_mode, cvals)

    face_c = np.nanmean(cvals[faces], axis=1)
    face_colors = cmap(norm(face_c))
    edgecolors = (0.3, 0.3, 0.3, 1.0)

    invalid = ~np.isfinite(face_c)
    face_c[invalid] = norm.vmin
    face_colors = cmap(norm(face_c))
    face_colors[invalid, 3] = 0.0

    mesh = Poly3DCollection(
        verts[faces],
        facecolors=face_colors,
        edgecolors=edgecolors,
        linewidths=0.15
    )
    mesh.set_alpha(1.0)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.add_collection3d(mesh)

    ax.set_xlim(0, nx - 1)
    ax.set_ylim(0, ny - 1)
    ax.set_zlim(0, nz - 1)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    try:
        ax.set_box_aspect((nx, ny, nz))
    except Exception:
        pass

    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(face_c)
    cbar = plt.colorbar(mappable, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label(_label_for_color_mode_3d(color_mode))

    plt.title(title + f" (iso={iso_level}, stride={stride})")
    plt.tight_layout()

    if out_dir is not None:
        filename = out_dir + title.replace(" ", "_") + '_' + color_mode + ".png"
        plt.savefig(filename, bbox_inches='tight')
        print(f"Phase Field image saved as {filename}!")
    
    plt.show()

def plot_phase_field_simulation(lattice: PhaseFieldLattice, out_dir: str,
                                color_field_name: str, field_name: str, title: str, three_dim: bool):
    """
    Wrapper function that decides whether to plot in 2D or 3D based on the lattice shape and user preference.

    Args:
        lattice (PhaseFieldLattice): custom lattice object.
        out_dir (str): output directory to save the plot. If None, the plot is not saved.
        color_field_name (str): name of the field to use for coloring.
        field_name (str): name of the field to plot.
        title (str): title of the plot.
        three_dim (bool): whether to plot in 3D or not.
    """
    mode_map = {
        'u': 'u',
        'curvature': 'curvature',
        'history': 'history',
        'phi': 'phi'
    }

    if not three_dim or (lattice.shape[2] == 1):
        mode = mode_map.get(color_field_name, 'phase')
        plot_2d_phase_field_simulation(lattice, out_dir, field_name, color_mode=mode, title=title)
        return
    else:
        mode = mode_map.get(color_field_name, 'phase')
        plot_3d_phase_field_simulation(lattice, out_dir, field_name, color_mode=mode, title=title)


# -----------------------------
# Kinetic Lattice helpers
# -----------------------------
def _build_id_palette(lattice: KineticLattice) -> tuple[np.ndarray, Colormap, dict]:
    """
    Returns the data grid, colormap, and ID-to-color mapping for ID-based coloring.

    Args:
        lattice (KineticLattice): custom lattice object.

    Returns:
        tuple[np.ndarray, Colormap, dict]: data grid, colormap, and ID-to-color mapping.
    """
    data_grid = lattice.group_id
    cmap = colormaps['tab20b']
    unique_vals = np.unique(data_grid[lattice.grid == 1])
    id_to_color = {uid: cmap(i % cmap.N) for i, uid in enumerate(unique_vals)}
    return data_grid, cmap, id_to_color

def _build_epoch_palette(lattice: KineticLattice, N_epochs: int) -> tuple[np.ndarray, Colormap, Normalize]:
    """
    Returns the data grid, colormap, and normalization for epoch-based coloring.

    Args:
        lattice (KineticLattice): custom lattice object.
        N_epochs (int): total number of epochs in the simulation.

    Returns:
        tuple[np.ndarray, Colormap, Normalize]: data grid, colormap, and normalization.
    """
    data_grid = lattice.history
    cmap = plt.cm.viridis
    norm = Normalize(vmin=0, vmax=N_epochs)
    return data_grid, cmap, norm

def _render_voxels_boundaries(ax: plt.Axes, lattice: KineticLattice, visible_voxels: np.ndarray) -> np.ndarray:
    """
    Renders the boundaries of grains in a 3D lattice visualization.

    Args:
        ax (plt.Axes): Matplotlib 3D axis.
        lattice (KineticLattice): custom lattice object.
        visible_voxels (np.ndarray): binary mask of visible voxels.
    """
    boundary_mask = get_grain_boundaries_mask(lattice)
    plot_mask = visible_voxels | boundary_mask

    colors = np.zeros(lattice.shape + (4,), dtype=np.float32)
    colors[visible_voxels] = (0.9, 0.9, 0.9, 0.1)
    colors[boundary_mask] = (0.0, 0.0, 0.0, 1.0)

    ax.voxels(plot_mask, facecolors=colors, edgecolor=None, linewidth=0)
    return colors

def _render_voxels_surface(ax: plt.Axes, lattice: KineticLattice, visible_voxels: np.ndarray, data_grid: np.ndarray,
                          color_mode: str, cmap: Colormap, norm_or_id_to_color: Normalize | dict) -> np.ndarray:
    """
    Renders the surface of voxels in a 3D lattice visualization based on the specified color mode.

    Args:
        ax (plt.Axes): Matplotlib 3D axis.
        lattice (KineticLattice): custom lattice object.
        visible_voxels (np.ndarray): binary mask of visible voxels.
        data_grid (np.ndarray): data grid for coloring.
        color_mode (str): "id" or "epoch".
        cmap (Colormap): colormap for epoch coloring.
        norm_or_id_to_color (Normalize | dict): normalization for epoch or ID-to-color mapping.

    Returns:
        np.ndarray: RGBA colors for the voxels.
    """
    colors = np.zeros(lattice.shape + (4,), dtype=np.float32)

    if np.any(visible_voxels):
        surface_values = data_grid[visible_voxels]
        unique_surface_vals = np.unique(surface_values)

        for val in unique_surface_vals:
            mask = (visible_voxels) & (data_grid == val)
            if color_mode == "id":
                id_to_color = norm_or_id_to_color
                colors[mask] = id_to_color[val]
            else:
                norm = norm_or_id_to_color
                colors[mask] = cmap(norm(val))

    return colors

def _plot_2d_epoch(ax: plt.Axes, lattice: KineticLattice, cmap: Colormap, norm: Normalize):
    """
    Plots a 2D representation of the lattice colored by epoch.
    Args:
        ax (plt.Axes): Matplotlib axis.
        lattice (KineticLattice): custom lattice object.
        cmap (Colormap): colormap for epoch coloring.
        norm (Normalize): normalization for epoch coloring.
    """
    data_2d = np.max(lattice.history, axis=2)
    masked_data = np.ma.masked_where(data_2d < 0, data_2d)
    im = ax.imshow(masked_data.T, origin='lower', cmap=cmap, norm=norm, interpolation='nearest')
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Occupation epoch")

def _plot_2d_id(ax: plt.Axes, lattice: KineticLattice, id_to_color: dict):
    """
    Plots a 2D representation of the lattice colored by grain ID.

    Args:
        ax (plt.Axes): Matplotlib axis.
        lattice (KineticLattice): custom lattice object.
        id_to_color (dict): mapping from grain ID to RGBA color.
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

def _plot_2d_boundaries(ax: plt.Axes, lattice: KineticLattice):
    """
    Plots a 2D representation of the lattice highlighting grain boundaries.

    Args:
        ax (plt.Axes): Matplotlib axis.
        lattice (KineticLattice): custom lattice object.
    """
    boundary_mask = get_grain_boundaries_mask(lattice)
    mask_2d = np.any(boundary_mask, axis=2)
    occ_2d = np.any(lattice.grid, axis=2)

    nx, ny = mask_2d.shape
    rgba_img = np.ones((ny, nx, 4))

    rgba_img[occ_2d.T] = [0.9, 0.9, 0.9, 1.0]
    rgba_img[mask_2d.T] = [0.0, 0.0, 0.0, 1.0]

    ax.imshow(rgba_img, origin='lower', interpolation='nearest')


# -----------------------------
# Kinetic Lattice plotting
# -----------------------------
def plot_kinetic_lattice(lattice: KineticLattice, N_epochs: int, title: str,
                 out_dir: str, three_dim: bool,
                 color_mode: str = "epoch"):
    """
    Function that creates a 2D or 3D plot of the kinetic lattice simulation.

    Args:
        lattice (KineticLattice): custom lattice object.
        N_epochs (int): total number of epochs in the simulation.
        title (str): title of the plot.
        out_dir (str): output directory to save the plot. If None, the plot is not saved.
        three_dim (bool): whether to plot in 3D or not.
        color_mode (str, optional): color mode ('epoch', 'id', 'boundaries'). Defaults to "epoch".
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

        if color_mode == "boundaries":
            colors = _render_voxels_boundaries(ax, lattice, visible_voxels)
        else:
            colors = _render_voxels_surface(
                ax=ax,
                lattice=lattice,
                visible_voxels=visible_voxels,
                data_grid=data_grid,
                color_mode=color_mode,
                cmap=cmap,
                norm_or_id_to_color=(id_to_color if color_mode == "id" else norm),
            )

        if color_mode in ["id", "boundaries"]:
            edge_color = None
            line_width = 0.0
        else:
            edge_color = 'k'
            line_width = 0.2

        ax.voxels(visible_voxels, facecolors=colors, edgecolor=None, linewidth=line_width)
        ax.set_title(title)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        nx, ny, nz = lattice.shape
        if len(lattice.occupied) > 0:
            occ_arr = np.array(list(lattice.occupied), dtype=int)
            max_y = int(np.max(occ_arr[:, 1])) if len(occ_arr) > 0 else ny
        else:
            max_y = 0
        ax.set_xlim(0, nx)
        ax.set_ylim(0, min(ny, max_y + 5))
        ax.set_zlim(0, nz)
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
        ax.set_xlabel('x')
        ax.set_ylabel('y')

    if color_mode == "id":
        legend_elements = [
            Patch(facecolor=c, edgecolor='k', label=f'ID: {int(uid)}')
            for uid, c in id_to_color.items()
        ]
        ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), title="Parameters")

    plt.tight_layout()

    if out_dir is not None:
        filename = out_dir + title.replace(" ", "_") + ".png"
        plt.savefig(filename, bbox_inches='tight')
        print(f"KineticLattice image saved as {filename}!")

    plt.show()
