import numpy as np
import logging
import pytest

from classes.GrowthModel import GrowthModel
from classes.EDENGrowth import EDENGrowth
from classes.DLAGrowth import DLAGrowth
from classes.KobayashiGrowth import KobayashiGrowth
from classes.StefanGrowth import StefanGrowth

# === Minimal STUB to isolate the tests =======================================
class FluxStub:
    def __init__(self, value=1.0):
        self.value = float(value)
        self.calls = 0

    def compute_external_flux_weights(self, direction_vec):
        self.calls += 1
        return self.value

class LatticeStubBase:
    """
    Common: GrowthModel.__str__/DLAGrowth.__str__ use len(lattice.occupied).
    """
    def __init__(self):
        self.occupied = set()

class EdenLatticeStub(LatticeStubBase):
    """
    Minimal case: only one active border and only one occupied neighbor with
    known gid.
    """
    def __init__(self):
        super().__init__()
        self.occupied = {(0, 0, 0)}
        self._gid = {(0, 0, 0): 7}
        self.last_occupy = None

    def get_active_border(self):
        return np.array([[1, 0, 0]], dtype=int)

    def get_crystal_bounding_box(self):
        return [(0, 1), (0, 0), (0, 0)]

    def get_neighbors(self, x, y, z):
        return np.array([[0, 0, 0]], dtype=int)

    def is_occupied(self, x, y, z):
        return (int(x), int(y), int(z)) in self.occupied

    def get_group_id(self, x, y, z):
        return int(self._gid[(int(x), int(y), int(z))])

    def occupy(self, x, y, z, epoch, id):
        self.last_occupy = (int(x), int(y), int(z), int(epoch), int(id))
        self.occupied.add((int(x), int(y), int(z)))

class EdenLatticeNoBorderStub(LatticeStubBase):
    """
    No border case.
    """
    def __init__(self):
        super().__init__()
        self.occupy_called = False

    def get_active_border(self):
        return np.array([]).reshape((0, 3))

    def get_crystal_bounding_box(self):
        return [(0, 0), (0, 0), (0, 0)]

    def get_neighbors(self, *args):
        return np.array([]).reshape((0, 3))

    def is_occupied(self, *args):
        return False

    def get_group_id(self, *args):
        return 0

    def occupy(self, *args, **kwargs):
        self.occupy_called = True

class DLALatticeStub(LatticeStubBase):
    """
    Dammy DLA object to test step() without real random walk.
    """
    def __init__(self):
        super().__init__()
        self.calls = []

    def get_crystal_bounding_box(self, padding=0):
        self.calls.append(int(padding))
        if padding == 1:
            return [(0, 1), (0, 1), (0, 1)]
        if padding == 2:
            return [(0, 2), (0, 2), (0, 2)]
        return [(0, 0), (0, 0), (0, 0)]

class PhaseFieldLatticeStub(LatticeStubBase):
    """
    Minimal object to test Kobayashi::step()
    """
    def __init__(self, nx=8, ny=8):
        super().__init__()
        self.phi = np.zeros((nx, ny, 1), dtype=np.float64)
        self.shape = self.phi.shape
        self.update_calls = []

    def update_occupied_and_history(self, epoch: int):
        self.update_calls.append(int(epoch))

class StefanLatticeStub:
    def __init__(self, nx=10, ny=10, dx=1.0):
        self.dx = dx
        self.phi = np.zeros((nx, ny), dtype=np.float64)
        self.u = np.zeros((nx, ny), dtype=np.float64)
        self.update_calls = []

    def update_occupied_and_history(self, epoch: int):
        self.update_calls.append(int(epoch))

class _CountingModel(GrowthModel):
    """
    Simple counter with minimal logs.
    """
    def __init__(self, lattice, **kw):
        super().__init__(lattice, **kw)
        self.calls = 0

    def step(self):
        self.calls += 1

# === Tests for the different growth models ===================================
def test_growthmodel_run_calls_step_and_increments_epoch(capsys):
    """
    Tests if the epochs are incremented whrn step is colled in
    any GrowthModel.
    """
    lat = LatticeStubBase()
    m = _CountingModel(lat, verbose=False)
    m.run(5)
    out, _ = capsys.readouterr()

    assert m.calls == 5
    assert m.epoch == 5

def test_eden_step_attaches_minimal_case():
    """
    Test a step in EDEN model.
    """
    lat = EdenLatticeStub()
    eden = EDENGrowth(lat, external_flux=None, rng_seed=0, verbose=False)

    eden.step()

    assert lat.last_occupy == (1, 0, 0, 0, 7)
    assert (1, 0, 0) in lat.occupied

def test_eden_step_no_active_border_skips(caplog):
    """
    Tests a step in EDEN model but with no active border.
    """
    lat = EdenLatticeNoBorderStub()
    eden = EDENGrowth(lat, external_flux=None, rng_seed=0, verbose=False)

    caplog.set_level(logging.WARNING, logger="growthsim")
    eden.step()

    assert "no active border" in caplog.text

def test_dla_init_validates_paddings():
    """
    Test the initiation of DLA growth.
    """
    lat = DLALatticeStub()
    with pytest.raises(ValueError, match="outer limit padding must be > generation padding"):
        DLAGrowth(lat, generation_padding=2, outer_limit_padding=2, verbose=False)

def test_dla_generate_random_point_on_box_is_on_surface_and_within_bounds():
    """
    Test the function DLAGrowth::generate_random_point_on_box()
    """
    lat = DLALatticeStub()
    dლა = DLAGrowth(lat, generation_padding=1, outer_limit_padding=2, rng_seed=0, verbose=False)

    box = [(0, 2), (10, 12), (5, 7)]
    p = dლა._generate_random_point_on_box(box)

    assert p.shape == (3,)
    for d in range(3):
        assert box[d][0] <= int(p[d]) <= box[d][1]
    assert sum(int(p[d]) in (box[d][0], box[d][1]) for d in range(3)) >= 1

def test_dla_step_calls_boxes_and_walk(monkeypatch):
    """
    Test a full step of DLA simulation.
    """
    lat = DLALatticeStub()
    dla = DLAGrowth(lat, generation_padding=1, outer_limit_padding=2, rng_seed=0, verbose=False)

    called = {}

    def fake_generate(box):
        called["gen_box"] = box
        return np.array([0, 0, 0], dtype=int)

    def fake_walk(start, outer_box):
        called["start"] = start
        called["outer_box"] = outer_box

    monkeypatch.setattr(dla, "_generate_random_point_on_box", fake_generate)
    monkeypatch.setattr(dla, "_particle_random_walk", fake_walk)

    dla.step()

    assert called["gen_box"] == [(0, 1), (0, 1), (0, 1)]
    assert np.array_equal(called["start"], np.array([0, 0, 0]))
    assert called["outer_box"] == [(0, 2), (0, 2), (0, 2)]
    assert lat.calls == [1, 2]

def test_kobayashi_step_updates_phi_and_calls_update():
    """
    Tests a full step of Kobayashi model.
    """
    lat = PhaseFieldLatticeStub(nx=10, ny=10)
    lat.phi[5, 5, 0] = 0.2

    kg = KobayashiGrowth(lat, dt=0.01, mobility=1.0, epsilon0=1.0, delta=0.0, n_folds=0.0,
                         supersaturation=0.0, three_dim=False, verbose=False)

    phi_before = lat.phi[:, :, 0].copy()
    kg.step()

    assert lat.update_calls == [0, 0]
    assert lat.phi[:, :, 0].shape == phi_before.shape
    assert not np.allclose(lat.phi[:, :, 0], phi_before)

    assert float(lat.phi[:, :, 0].min()) >= -1e-3 - 1e-12
    assert float(lat.phi[:, :, 0].max()) <= 1.0 + 1e-3 + 1e-12

def test_stefan_init_sets_tau_and_references_fields():
    lat = StefanLatticeStub(nx=10, ny=10, dx=2.0)
    sg = StefanGrowth(lat, mobility=2.0, verbose=False)

    assert np.isclose(sg.tau, 0.5)
    assert sg.dx == 2.0
    assert sg.phi is lat.phi
    assert sg.u is lat.u

def test_stefan_init_mobility_zero_uses_default_tau():
    lat = StefanLatticeStub()
    sg = StefanGrowth(lat, mobility=0.0, verbose=False)
    assert np.isclose(sg.tau, 0.0003)

def test_stefan_step_updates_fields_and_calls_update():
    lat = StefanLatticeStub(nx=12, ny=12, dx=1.0)

    lat.phi[5, 5] = 0.2

    sg = StefanGrowth(lat, dt=0.01, mobility=1.0, delta=0.0, verbose=False)

    phi_before = lat.phi.copy()
    u_before = lat.u.copy()

    sg.step()

    assert lat.update_calls == [0]
    assert not np.allclose(lat.phi, phi_before)
    assert not np.allclose(lat.u, u_before)
    assert float(lat.phi.min()) >= 0.0
    assert float(lat.phi.max()) <= 1.0

def test_stefan_step_noise_only_every_10_epochs(monkeypatch):
    lat = StefanLatticeStub(nx=12, ny=12, dx=1.0)
    lat.phi[5, 5] = 0.2
    sg = StefanGrowth(lat, dt=0.01, mobility=1.0, delta=0.0, verbose=False)

    calls = {"n": 0}

    def fake_normal(*args, **kwargs):
        calls["n"] += 1
        return np.zeros((10, 10), dtype=np.float64)

    monkeypatch.setattr(np.random, "normal", fake_normal)

    sg.epoch = 0
    sg.step()
    assert calls["n"] == 1

    sg.epoch = 1
    sg.step()
    assert calls["n"] == 1
