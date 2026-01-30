import os
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pytest

import matplotlib.pyplot as plt

from utils.GUI import (
    get_visible_voxels_binary_mask,
    get_grain_boundaries_mask,
    _mid_plane_z,
    compute_curvature_2d,
    get_field_3d,
    _get_data_2d_by_name,
    _cmap_name_for_mode,
    _norm_for_3d_mode,
    _default_iso_level,
    _build_id_palette,
    _build_epoch_palette,
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
        self.curvature = curvature if curvature is not None else np.zeros_like(phi)
        self.history = history if history is not None else (np.ones_like(phi, dtype=np.int64) * (-1))
        self.shape = phi.shape


# -----------------------
# Tests for utility functions
# -----------------------
def test_mid_plane_z():
    assert _mid_plane_z(type("L", (), {"shape": (3, 3, 1)})()) == 0
    assert _mid_plane_z(type("L", (), {"shape": (3, 3, 5)})()) == 2

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


def test_compute_curvature_2d_constant_field_is_zeroish():
    phi = np.ones((10, 10), dtype=float) * 0.3
    curv = compute_curvature_2d(phi)
    assert curv.shape == phi.shape
    assert np.allclose(curv, 0.0, atol=1e-6)

def test_get_field_3d_returns_attr_or_raises():
    phi = np.zeros((4, 4, 1), dtype=np.float32)
    lat = PhaseFieldStub(phi=phi)

    got = get_field_3d(lat, "phi")
    assert got is lat.phi

    with pytest.raises(ValueError, match="has no field"):
        get_field_3d(lat, "does_not_exist")

def test_get_data_2d_by_name_known_and_unknown(capsys):
    phi = np.zeros((4, 4, 3), dtype=np.float32)
    phi[1, 2, 1] = 7.0
    lat = PhaseFieldStub(phi=phi)

    mid = 1
    a = _get_data_2d_by_name(lat, "phi", mid)
    assert a.shape == (4, 4)
    assert a[1, 2] == 7.0

    b = _get_data_2d_by_name(lat, "unknown", mid)
    out, _ = capsys.readouterr()
    assert b is None
    assert "Unknown field" in out

def test_cmap_name_for_mode_minimal():
    assert _cmap_name_for_mode(None) == "viridis"
    assert _cmap_name_for_mode("phi") == "gray_r"
    assert _cmap_name_for_mode("history") == "turbo"

def test_norm_for_3d_mode_history_phi_curvature():
    phi = np.zeros((4, 4, 2), dtype=np.float32)
    hist = np.ones((4, 4, 2), dtype=np.int64) * (-1)
    hist[0, 0, 0] = 5
    hist[1, 1, 1] = 10
    lat = PhaseFieldStub(phi=phi, history=hist)

    n = _norm_for_3d_mode(lat, "history", cvals=np.array([0.0, 1.0]))
    assert n.vmin == 5.0
    assert n.vmax == 10.0

    n2 = _norm_for_3d_mode(lat, "phi", cvals=np.array([0.2, 0.7]))
    assert n2.vmin == 0.0
    assert n2.vmax == 1.0

    n3 = _norm_for_3d_mode(lat, "curvature", cvals=np.array([-2.0, 1.0, 0.0]))
    assert n3.vmin <= 0.0 <= n3.vmax
    assert np.isclose(abs(n3.vmin), abs(n3.vmax), rtol=1e-6)

def test_default_iso_level():
    vol = np.array([[[0.0, 0.2], [0.1, 0.4]]], dtype=np.float32)
    assert np.isclose(_default_iso_level(vol), 0.2)

    vol2 = np.array([[[0.0, 0.8]]], dtype=np.float32)
    assert np.isclose(_default_iso_level(vol2), 0.5)

def test_build_palettes_minimal():
    grid = np.zeros((3, 3, 1), dtype=np.uint8)
    grid[0, 0, 0] = 1
    gid = np.zeros_like(grid, dtype=np.uint16)
    gid[0, 0, 0] = 9
    hist = np.ones_like(grid, dtype=np.int64) * (-1)
    hist[0, 0, 0] = 3

    lat = KineticStub(grid=grid, group_id=gid, history=hist)

    data_grid, cmap, id_to_color = _build_id_palette(lat)
    assert data_grid is gid
    assert 9 in id_to_color

    data_grid2, cmap2, norm2 = _build_epoch_palette(lat, N_epochs=10)
    assert data_grid2 is hist
    assert norm2.vmin == 0
    assert norm2.vmax == 10


# -----------------------
# Smoke test plotting
# -----------------------
def test_plot_kinetic_lattice_invalid_color_mode_returns(capsys, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)

    grid = np.zeros((3, 3, 1), dtype=np.uint8)
    lat = KineticStub(grid=grid)

    plot_kinetic_lattice(lat, N_epochs=10, title="t", out_dir=None, three_dim=False, color_mode="bad")
    out, _ = capsys.readouterr()
    assert "only accepted" in out
