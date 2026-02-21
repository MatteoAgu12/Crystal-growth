import os
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pytest
import logging

import matplotlib.pyplot as plt

from GUI.GUI import (
    get_visible_voxels_binary_mask,
    get_grain_boundaries_mask,
    plot_kinetic_lattice,
)


# -----------------------
# Minimal stub classes
# -----------------------
class KineticStub:
    def __init__(self, grid, group_id=None, history=None, occupied=None):
        self.grid = grid
        self.group_id = group_id if group_id is not None else np.zeros_like(grid, dtype=np.uint16)
        self.history = history if history is not None else (np.ones_like(grid, dtype=np.int64) * (-1))
        self.shape = grid.shape
        self.occupied = occupied if occupied is not None else set()

class PhaseFieldStub:
    def __init__(self, phi, u=None, curvature=None, history=None):
        self.phi = phi
        self.u = u if u is not None else np.zeros_like(phi)
        self.history = history if history is not None else (np.ones_like(phi, dtype=np.int64) * (-1))
        self.shape = phi.shape


# -----------------------
# Tests for utility functions
# -----------------------
def test_get_visible_voxels_binary_mask_single_voxel_visible():
    grid = np.zeros((3, 3, 3), dtype=np.uint8)
    grid[1, 1, 1] = 1
    lat = KineticStub(grid=grid)
    visible = get_visible_voxels_binary_mask(lat)
    assert visible.shape == grid.shape
    assert visible[1, 1, 1]    

def test_get_visible_voxels_binary_mask_hides_fully_surrounded_interior():
    grid = np.ones((3, 3, 3), dtype=np.uint8)
    lat = KineticStub(grid=grid)
    visible = get_visible_voxels_binary_mask(lat)

    assert not visible[1, 1, 1]
    assert visible[0, 1, 1]

def test_get_grain_boundaries_mask_marks_interfaces_only_for_occupied():
    gid = np.zeros((3, 3, 1), dtype=np.uint16)
    gid[0, 0, 0] = 1
    gid[1, 0, 0] = 2

    lat = type("L", (), {"group_id": gid, "shape": gid.shape})()
    mask = get_grain_boundaries_mask(lat)

    assert mask.shape == gid.shape
    assert mask[0, 0, 0] 
    assert mask[1, 0, 0] 
    assert not mask[2, 2, 0] 

# -----------------------
# Smoke test plotting
# -----------------------
def test_plot_kinetic_lattice_invalid_color_mode_returns(caplog, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)

    grid = np.zeros((3, 3, 1), dtype=np.uint8)
    lat = KineticStub(grid=grid)

    plot_kinetic_lattice(lat, N_epochs=10, title="t", out_dir=None, three_dim=False, color_mode="bad")
    caplog.set_level(logging.WARNING, logger="growthsim")
    assert "only accepted" in caplog.text
