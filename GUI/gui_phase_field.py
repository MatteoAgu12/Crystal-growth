import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
from scipy.ndimage import map_coordinates
from typing import Union

from classes.PhaseFieldLattice import PhaseFieldLattice
from GUI.gui_routines import _mid_plane_z, set_axes_labels, finalize_plot

import logging
logger = logging.getLogger("growthsim")

def get_field_3d(lattice: PhaseFieldLattice, field_name: str) -> np.ndarray:
    """
    Determine which continuous field to analyze.

    Args:
        lattice (PhaseFieldLattice): custom lattice object for phase field simulations
        field_name (str): name of the requested field

    Raise:
        ValueError: if the lattice has not that field.

    Return:
        (np.ndarray): 3D array of the requested field
    """
    if not hasattr(lattice, field_name):
        raise ValueError(f"[GUI] lattice has no field '{field_name}'")
    return getattr(lattice, field_name)

def _get_data_2d_by_name(lattice: PhaseFieldLattice, field_name: str, mid_z: int) -> Union[np.ndarray, None]:
    """
    Retrieve the 2D data slice from the lattice based on the field name.

    Args:
        lattice (PhaseFieldLattice): custom lattice object for phase field simulations
        field_name (str): name of the requested field
        mid_z (int): z index of the mid-plane to slice

    Return:
        (np.ndarray or None): 2D array slice of the requested field, or None if unknown
    """
    if field_name == 'phi':
        return lattice.phi[:, :, mid_z]
    if field_name == 'u':
        return lattice.u[:, :, mid_z]
    if field_name == 'history':
        return lattice.history[:, :, mid_z]
    logger.warning(f"[GUI] Error: Unknown field {field_name}")
    return None

def _cmap_name_for_mode(color_mode: Union[str, None]) -> str:
    """
    Return the colormap name for the given color mode.

    Args:
        color_mode (str or None): name of the color mode

    Return:
        (str): string identifier for the matplotlib colormap
    """
    if color_mode is not None:
        if color_mode == 'phi': return 'gray_r'
        if color_mode == 'u': return 'inferno'
        if color_mode == 'history': return 'turbo'
    return 'turbo'

def _norm_for_3d_mode(lattice: PhaseFieldLattice, color_mode: str, cvals: np.ndarray) -> Normalize:
    """
    Return the Normalize object for the given color mode in 3D plotting.

    Args:
        lattice (PhaseFieldLattice): custom lattice object
        color_mode (str): color mode identifier
        cvals (np.ndarray): array of color values extracted from the volume

    Return:
        (Normalize): matplotlib normalization object bounding the color limits
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
    else:
        vmin, vmax = None, None

    return Normalize(vmin=vmin, vmax=vmax)

def _label_for_color_mode_3d(color_mode: str) -> str:
    """
    Return the colorbar label text for the given color mode in 3D plotting.

    Args:
        color_mode (str): color mode identifier

    Return:
        (str): formatted string for the colorbar label
    """
    if color_mode == 'phi': return "Phase Field $\\phi$"
    if color_mode == 'u': return "Field $u$"
    if color_mode == 'history': return "Solidification Epoch"
    return str(color_mode)

def _default_iso_level(vol: np.ndarray) -> float:
    """
    Determine a default iso-level for the marching cubes algorithm based on the volume data.

    Args:
        vol (np.ndarray): 3D volume data array

    Return:
        (float): computed default iso-level
    """
    vmin = float(np.nanmin(vol))
    vmax = float(np.nanmax(vol))
    return vmin + 0.5 * (vmax - vmin) if vmax < 0.5 else 0.5

def plot_2d_phase_field_simulation(lattice: PhaseFieldLattice, out_dir: Union[str, None], 
                                   field_name: str, color_mode: str, title: str) -> None:
    """
    Create and display a 2D plot of the phase field simulation slice.

    Args:
        lattice (PhaseFieldLattice): custom lattice object
        out_dir (str or None): output directory to save the plot
        field_name (str): name of the field to plot
        color_mode (str): color mode identifier ('phi', 'u', 'history')
        title (str): title of the plot
    """
    mid_z = _mid_plane_z(lattice)
    data_2d = _get_data_2d_by_name(lattice, field_name, mid_z)
    if data_2d is None: return

    fig, ax = plt.subplots(figsize=(10, 8))

    if color_mode == 'phi':
        im = ax.imshow(data_2d.T, origin='lower', cmap='gray_r', vmin=0, vmax=1)
        label = r"Phase Field $\phi$"
    elif color_mode == 'u':
        im = ax.imshow(data_2d.T, origin='lower', cmap='inferno', vmin=-1.0, vmax=1.0)
        label = r"Diffused field $u$"
    elif color_mode == 'history':
        hist = lattice.history[:, :, mid_z].astype(float)
        hist_masked = np.ma.masked_where(hist < 0, hist)
        im = ax.imshow(hist_masked.T, origin='lower', cmap='turbo')
        label = "Solidification Epoch"

    plt.colorbar(im, ax=ax, label=label)
    ax.set_title(f"{title} - {color_mode.capitalize()}")
    set_axes_labels(ax, is_3d=False)
    
    finalize_plot(out_dir, title, color_mode, "Phase Field image saved as %s!")

def _plot_3d_phase_field_simulation(lattice: PhaseFieldLattice, out_dir: Union[str, None], 
                                    field_name: str, color_mode: str, title: str, iso_level: Union[float, None] = None) -> None:
    """
    Create a 3D plot of the phase field simulation using marching cubes.

    Args:
        lattice (PhaseFieldLattice): custom lattice object
        out_dir (str or None): output directory to save the plot
        field_name (str): name of the field to plot
        color_mode (str): color mode identifier ('phi', 'u', 'history')
        title (str): title of the plot
        iso_level (float or None): iso-level for the marching cubes algorithm
    """
    phi = lattice.phi
    nx, ny, nz = phi.shape
    stride = 1

    phi_ds = phi[::stride, ::stride, ::stride]
    vol = phi_ds.astype(np.float64, copy=False)

    if iso_level is None:
        iso_level = _default_iso_level(vol)

    try:
        verts, faces, normals, values = measure.marching_cubes(
            vol, level=float(iso_level), spacing=(stride, stride, stride)
        )
    except Exception as e:
        logger.warning("[GUI] marching_cubes failed: %s", e)
        return

    if color_mode in ('phi', 'u', 'history'):
        cfield = get_field_3d(lattice, color_mode)[::stride, ::stride, ::stride]
    else:
        logger.warning("[GUI] Unknown color_mode='%s', Return...", color_mode)
        return

    coords = np.vstack([verts[:, 0], verts[:, 1], verts[:, 2]])
    cvals = map_coordinates(cfield.astype(np.float64, copy=False), coords, order=1, mode='nearest')

    if color_mode == 'history':
        cvals = np.where(cvals >= 0, cvals, np.nan)

    cmap = cm.get_cmap(_cmap_name_for_mode(color_mode))
    norm = _norm_for_3d_mode(lattice, color_mode, cvals)

    face_c = np.nanmean(cvals[faces], axis=1)
    invalid = ~np.isfinite(face_c)
    face_c[invalid] = norm.vmin
    
    face_colors = cmap(norm(face_c))
    face_colors[invalid, 3] = 0.0

    mesh = Poly3DCollection(verts[faces], facecolors=face_colors, edgecolors=(0.3, 0.3, 0.3, 1.0), linewidths=0.15)
    
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.add_collection3d(mesh)

    ax.set_xlim(0, nx - 1); ax.set_ylim(0, ny - 1); ax.set_zlim(0, nz - 1)
    set_axes_labels(ax, is_3d=True)

    try: ax.set_box_aspect((nx, ny, nz))
    except Exception: pass

    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(face_c)
    plt.colorbar(mappable, ax=ax, shrink=0.7, pad=0.02).set_label(_label_for_color_mode_3d(color_mode))
    ax.set_title(f"{title} (iso={iso_level}, stride={stride})")

    finalize_plot(out_dir, title, color_mode, "Phase Field image saved as %s!")

def plot_phase_field_simulation(lattice: PhaseFieldLattice, out_dir: Union[str, None], 
                                color_field_name: str, field_name: str, title: str, three_dim: bool) -> None:
    """
    Wrapper function that decides whether to plot in 2D or 3D based on the lattice shape and user preference.

    Args:
        lattice (PhaseFieldLattice): custom lattice object
        out_dir (str or None): output directory to save the plot
        color_field_name (str): name of the field to use for coloring
        field_name (str): name of the field to extract and plot
        title (str): title of the plot
        three_dim (bool): flag to toggle 3D plotting
    """
    mode_map = {'u': 'u', 'history': 'history', 'phi': 'phi'}

    if not three_dim or (lattice.shape[2] == 1):
        mode = mode_map.get(color_field_name, 'phase')
        plot_2d_phase_field_simulation(lattice, out_dir, field_name, color_mode=mode, title=title)
    else:
        return