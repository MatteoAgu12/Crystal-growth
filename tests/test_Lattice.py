import numpy as np
import pytest

from classes.Lattice import Lattice
from classes.BaseLattice import BaseLattice
from classes.PhaseFieldLattice import PhaseFieldLattice

@pytest.fixture
def dammy_lattice(capsys):
    lat = Lattice(3, 4, 5)
    capsys.readouterr()
    return lat

def test_init_rejects_negative_dimensions():
    with pytest.raises(ValueError, match="Lattice dimensions must be >= 0"):
        Lattice(-1, 1, 1, verbose=False)

def test_init_sets_grid_and_inherited_arrays(dammy_lattice):
    lat = dammy_lattice
    assert lat.shape == (3, 4, 5)

    assert isinstance(lat.grid, np.ndarray)
    assert lat.grid.shape == (3, 4, 5)
    assert lat.grid.dtype == np.uint8
    assert np.all(lat.grid == 0)

    # ereditati da BaseLattice
    assert lat.history.shape == (3, 4, 5)
    assert lat.history.dtype == np.int64
    assert np.all(lat.history == -1)

    assert lat.group_id.shape == (3, 4, 5)
    assert lat.group_id.dtype == np.uint16
    assert np.all(lat.group_id == 0)

    assert lat.group_counter == 0
    assert lat.initial_seeds == []
    assert lat.verbose is False

def test_is_point_inside_inherited(dammy_lattice):
    lat = dammy_lattice
    assert lat.is_point_inside(0, 0, 0) is True
    assert lat.is_point_inside(2, 3, 4) is True
    assert lat.is_point_inside(3, 0, 0) is False
    assert lat.is_point_inside(-1, 0, 0) is False

def test_get_neighbors_inherited(dammy_lattice):
    lat = dammy_lattice
    neigh = lat.get_neighbors(1, 1, 1)
    assert isinstance(neigh, np.ndarray)
    assert neigh.shape == (6, 3)

    got = {tuple(map(int, row)) for row in neigh}
    expected = {
        (2, 1, 1), (0, 1, 1),
        (1, 2, 1), (1, 0, 1),
        (1, 1, 2), (1, 1, 0),
    }
    assert got == expected

def test_occupy_outside_does_nothing(dammy_lattice):
    lat = dammy_lattice

    before_grid = lat.grid.copy()
    before_hist = lat.history.copy()
    before_gid = lat.group_id.copy()

    lat.occupy(99, 0, 0, epoch=0, id=7)

    assert np.array_equal(lat.grid, before_grid)
    assert np.array_equal(lat.history, before_hist)
    assert np.array_equal(lat.group_id, before_gid)
    assert len(lat.occupied) == 0

def test_occupy_sets_grid_history_group_and_occupied(dammy_lattice):
    lat = dammy_lattice
    lat.occupy(1, 2, 3, epoch=5, id=12)

    assert lat.grid[1, 2, 3] == 1
    assert lat.history[1, 2, 3] == 5
    assert lat.group_id[1, 2, 3] == 12
    assert (1, 2, 3) in lat.occupied

def test_occupy_twice_same_cell_does_not_change_epoch_or_id(dammy_lattice):
    lat = dammy_lattice
    lat.occupy(1, 1, 1, epoch=2, id=3)
    lat.occupy(1, 1, 1, epoch=99, id=99)

    assert lat.grid[1, 1, 1] == 1
    assert lat.history[1, 1, 1] == 2
    assert lat.group_id[1, 1, 1] == 3
    assert len(lat.occupied) == 1

def test_is_occupied(dammy_lattice):
    lat = dammy_lattice
    assert lat.is_occupied(0, 0, 0) is False

    lat.occupy(0, 0, 0, epoch=0, id=1)
    assert lat.is_occupied(0, 0, 0) is True
    assert lat.is_occupied(0.0, 0.0, 0.0) is True


def test_get_group_id(dammy_lattice):
    lat = dammy_lattice
    lat.occupy(2, 3, 4, epoch=1, id=42)
    assert lat.get_group_id(2, 3, 4) == 42


def test_get_group_counter(dammy_lattice):
    lat = dammy_lattice
    assert lat.get_group_counter() == 0
    lat.group_counter = 10
    assert lat.get_group_counter() == 10

def test_set_nucleation_seed_increments_group_counter_and_occupies(dammy_lattice):
    lat = dammy_lattice
    lat.set_nucleation_seed(1, 1, 1)

    assert lat.group_counter == 1
    assert (1, 1, 1) in lat.initial_seeds
    assert lat.is_occupied(1, 1, 1) is True
    assert lat.history[1, 1, 1] == 0
    assert lat.group_id[1, 1, 1] == 1


def test_set_nucleation_seed_duplicate_does_nothing(dammy_lattice):
    lat = dammy_lattice
    lat.set_nucleation_seed(1, 1, 1)
    lat.set_nucleation_seed(1, 1, 1)

    assert lat.group_counter == 1
    assert lat.initial_seeds.count((1, 1, 1)) == 1
    assert len(lat.occupied) == 1

def test_set_nucleation_seed_maintain_last_id(dammy_lattice):
    lat = dammy_lattice

    lat.set_nucleation_seed(0, 0, 0)  # group_counter -> 1
    lat.set_nucleation_seed(0, 0, 1, maintain_last_id=True)  # non incrementa

    assert lat.group_counter == 1
    assert lat.group_id[0, 0, 0] == 1
    assert lat.group_id[0, 0, 1] == 1


def test_get_nucleation_seeds(dammy_lattice):
    lat = dammy_lattice
    lat.set_nucleation_seed(0, 0, 0)
    lat.set_nucleation_seed(2, 3, 4)

    seeds = lat.get_nucleation_seeds()
    assert isinstance(seeds, np.ndarray)
    assert seeds.shape == (2, 3)
    assert {tuple(row) for row in seeds} == {(0, 0, 0), (2, 3, 4)}

def test_get_crystal_bounding_box_none_when_empty(dammy_lattice):
    lat = dammy_lattice
    assert lat.get_crystal_bounding_box() is None

def test_get_crystal_bounding_box_basic(dammy_lattice):
    lat = dammy_lattice
    lat.occupy(1, 1, 1, epoch=0, id=1)
    lat.occupy(2, 3, 4, epoch=0, id=1)

    box = lat.get_crystal_bounding_box(padding=0)
    assert box == [(1, 2), (1, 3), (1, 4)]


def test_get_crystal_bounding_box_with_padding_and_clipping(dammy_lattice):
    lat = dammy_lattice
    lat.occupy(0, 0, 0, epoch=0, id=1)
    lat.occupy(2, 3, 4, epoch=0, id=1)

    box = lat.get_crystal_bounding_box(padding=10)
    assert box == [(0, 2), (0, 3), (0, 4)]

@pytest.fixture
def dammy_pfl(capsys):
    lat = PhaseFieldLattice(7, 7, 2, interface_threshold=0.5, verbose=False)
    capsys.readouterr()
    return lat

def test_is_occupied_reflects_grid(dammy_pfl):
    lat = dammy_pfl
    assert lat.is_occupied(1, 1, 0) is False
    lat.grid[1, 1, 0] = 1
    assert lat.is_occupied(1, 1, 0) is True

def test_update_occupied_and_history_sets_grid_and_occupied(dammy_pfl):
    lat = dammy_pfl

    lat.phi[2, 3, 1] = 1.0
    lat.update_occupied_and_history(epoch=7)

    assert lat.grid[2, 3, 1] == 1
    assert (2, 3, 1) in lat.occupied
    assert lat.history[2, 3, 1] == 7  
    
    lat.phi[2, 3, 1] = 0.0
    lat.update_occupied_and_history(epoch=8)
    assert lat.grid[2, 3, 1] == 0
    assert (2, 3, 1) not in lat.occupied

    lat.phi[2, 3, 1] = 1.0
    lat.update_occupied_and_history(epoch=9)
    assert lat.grid[2, 3, 1] == 1
    assert lat.history[2, 3, 1] == 7

def test_set_nucleation_seed_updates_phi_and_registers_seed(dammy_pfl):
    lat = dammy_pfl
    z = 0

    lat.set_nucleation_seed(x=3, y=3, z=z, radius=2.0, width=0.5, phi_in=1.0, phi_out=0.0)

    # seed registrato
    assert (3, 3, z) in lat._seeds

    # phi slice aggiornato e non tutto zero
    assert lat.phi[:, :, z].max() > 0.0

    # grid/occupied aggiornati in base alla soglia
    assert np.any(lat.grid[:, :, z] == 1)
    assert len(lat.occupied) > 0

    # con un solo seed, i group_id assegnati ai nuovi occupati dovrebbero essere 0 (indice seed)
    # (nota: è il comportamento del tuo _nearest_seed_id)
    occ = list(lat.occupied)
    x0, y0, z0 = occ[0]
    assert lat.group_id[x0, y0, z0] == 0

    
def test_nearest_seed_id_prefers_closest(dammy_pfl):
    lat = dammy_pfl

    # due seed lontani
    lat.set_nucleation_seed(1, 1, 0, radius=1.0, width=0.5)
    lat.set_nucleation_seed(5, 5, 0, radius=1.0, width=0.5)

    # forza una cella vicino al secondo seed ad essere occupata
    lat.phi[5, 5, 0] = 1.0
    lat.update_occupied_and_history(epoch=1)

    assert (5, 5, 0) in lat.occupied
    assert lat.group_id[5, 5, 0] == 1  # indice del secondo seed