"""
This file is a wrapper for the main graphical features
and their visualization in 2D and in 3D.

Its function is exporting only the accessible functions when this file is imported.
The accessible functions are:
    - plot_kinetic_lattice
    - plot_phase_field_simulation
    - get_grain_boundaries_mask
    - get_visible_voxels_binary_mask
"""
from GUI.gui_routines import create_gif

from GUI.gui_kinetic import (
    plot_kinetic_lattice, 
    get_grain_boundaries_mask, 
    get_visible_voxels_binary_mask
)

from GUI.gui_phase_field import (
    plot_phase_field_simulation
)

__all__ = [
    "create_gif"
    "plot_kinetic_lattice",
    "plot_phase_field_simulation",
    "get_grain_boundaries_mask",
    "get_visible_voxels_binary_mask"
]