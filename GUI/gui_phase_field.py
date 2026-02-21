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