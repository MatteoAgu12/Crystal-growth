import numpy as np
import pytest

from classes.KineticLattice import KineticLattice
from classes.BaseLattice import BaseLattice
from classes.PhaseFieldLattice import PhaseFieldLattice


# === Creating dummy objects for the following tests ==========================
@pytest.fixture
def dammy_lattice(capsys):
    """
    Creates a dummy KineticLattice object
    """
    lat = KineticLattice(3, 4, 5)
    capsys.readouterr()
    return lat

@pytest.fixture
def dammy_pfl(capsys):
    """
    Creates a dammy PhaseFieldLattice object
    """
    lat = PhaseFieldLattice(7, 7, 2, interface_threshold=0.5, verbose=False)
    capsys.readouterr()
    return lat

# === Testing KineticLattice ==================================================
def test_init_rejects_negative_dimensions():
    """
    Checsks if the class raises an error if the size in input is negative
    """
    with pytest.raises(ValueError, match="KineticLattice dimensions must be >= 0"):
        KineticLattice(-1, 1, 1, verbose=False)

def test_init_sets_grid_and_inherited_arrays(dammy_lattice):
    """
    Checks that the grid is initialized correctly.
    Also checks that all inherited objects from BaseLattice are initialized correctly.
    """
    lat = dammy_lattice
    assert lat.shape == (3, 4, 5)

    assert isinstance(lat.grid, np.ndarray)
    assert lat.grid.shape == (3, 4, 5)
    assert lat.grid.dtype == np.uint8
    assert np.all(lat.grid == 0)

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
    """
    Tests the function BaseLattice::is_point_inside().
    """
    lat = dammy_lattice
    assert lat.is_point_inside(0, 0, 0) is True
    assert lat.is_point_inside(2, 3, 4) is True
    assert lat.is_point_inside(3, 0, 0) is False
    assert lat.is_point_inside(-1, 0, 0) is False

def test_get_neighbors_inherited(dammy_lattice):
    """
    Tests the function BaseLattice::get_neighbors().
    """
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
    """
    Test if set the occupancy of a cell not in the lattice causes a crash,
    nor modify any member.
    """
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
    """
    Checks the occupancy, the history and the id objects.
    """
    lat = dammy_lattice
    lat.occupy(1, 2, 3, epoch=5, id=12)

    assert lat.grid[1, 2, 3] == 1
    assert lat.history[1, 2, 3] == 5
    assert lat.group_id[1, 2, 3] == 12
    assert (1, 2, 3) in lat.occupied

def test_occupy_twice_same_cell_does_not_change_epoch_or_id(dammy_lattice):
    """
    Asserts that occupy a cell already occupied does nothing
    nor doesn't change the cell id.
    """
    lat = dammy_lattice
    lat.occupy(1, 1, 1, epoch=2, id=3)
    lat.occupy(1, 1, 1, epoch=99, id=99)

    assert lat.grid[1, 1, 1] == 1
    assert lat.history[1, 1, 1] == 2
    assert lat.group_id[1, 1, 1] == 3
    assert len(lat.occupied) == 1

def test_is_occupied(dammy_lattice):
    """
    Tests the function KineticLattice::is_occupied()
    """
    lat = dammy_lattice
    assert lat.is_occupied(0, 0, 0) is False

    lat.occupy(0, 0, 0, epoch=0, id=1)
    assert lat.is_occupied(0, 0, 0) is True
    assert lat.is_occupied(0.0, 0.0, 0.0) is True

def test_get_group_id(dammy_lattice):
    """
    Tests the function KineticLattice::get_group_id()
    """
    lat = dammy_lattice
    lat.occupy(2, 3, 4, epoch=1, id=42)
    assert lat.get_group_id(2, 3, 4) == 42

def test_get_group_counter(dammy_lattice):
    """
    Tests the function KineticLattice::get_group_counter()
    """
    lat = dammy_lattice
    assert lat.get_group_counter() == 0
    lat.group_counter = 10
    assert lat.get_group_counter() == 10

def test_set_nucleation_seed_increments_group_counter_and_occupies(dammy_lattice):
    """
    Checks that occupy a new cell modifies all the objects correctly.
    """
    lat = dammy_lattice
    lat.set_nucleation_seed(1, 1, 1)

    assert lat.group_counter == 1
    assert (1, 1, 1) in lat.initial_seeds
    assert lat.is_occupied(1, 1, 1) is True
    assert lat.history[1, 1, 1] == 0
    assert lat.group_id[1, 1, 1] == 1

def test_set_nucleation_seed_duplicate_does_nothing(dammy_lattice):
    """
    Checks that setting the same cell as a nucleation seed does nothing.
    """
    lat = dammy_lattice
    lat.set_nucleation_seed(1, 1, 1)
    lat.set_nucleation_seed(1, 1, 1)

    assert lat.group_counter == 1
    assert lat.initial_seeds.count((1, 1, 1)) == 1
    assert len(lat.occupied) == 1

def test_set_nucleation_seed_maintain_last_id(dammy_lattice):
    """
    Checks that the flag maintain_last_id works correctly.
    """
    lat = dammy_lattice

    lat.set_nucleation_seed(0, 0, 0)
    lat.set_nucleation_seed(0, 0, 1, maintain_last_id=True)

    assert lat.group_counter == 1
    assert lat.group_id[0, 0, 0] == 1
    assert lat.group_id[0, 0, 1] == 1

def test_get_nucleation_seeds(dammy_lattice):
    """
    Tests the function KineticLattice::get_nucleation_seeds()
    """
    lat = dammy_lattice
    lat.set_nucleation_seed(0, 0, 0)
    lat.set_nucleation_seed(2, 3, 4)

    seeds = lat.get_nucleation_seeds()
    assert isinstance(seeds, np.ndarray)
    assert seeds.shape == (2, 3)
    assert {tuple(row) for row in seeds} == {(0, 0, 0), (2, 3, 4)}

def test_get_crystal_bounding_box_none_when_empty(dammy_lattice):
    """
    Checks that the function KineticLattice::get_crystal_bounding_box()
    returns None when no cells are occupied.
    """
    lat = dammy_lattice
    assert lat.get_crystal_bounding_box() is None

def test_get_crystal_bounding_box_basic(dammy_lattice):
    """
    Tests the function KineticLattice::get_crystal_bounding_box()
    """
    lat = dammy_lattice
    lat.occupy(1, 1, 1, epoch=0, id=1)
    lat.occupy(2, 3, 4, epoch=0, id=1)

    box = lat.get_crystal_bounding_box(padding=0)
    assert box == [(1, 2), (1, 3), (1, 4)]

def test_get_crystal_bounding_box_with_padding_and_clipping(dammy_lattice):
    """
    Tests the function KineticLattice::get_crystal_bounding_box()
    also using padding.
    """
    lat = dammy_lattice
    lat.occupy(0, 0, 0, epoch=0, id=1)
    lat.occupy(2, 3, 4, epoch=0, id=1)

    box = lat.get_crystal_bounding_box(padding=10)
    assert box == [(0, 2), (0, 3), (0, 4)]

# === Testing PhaseFieldLattice =============================================== 
def test_is_occupied_reflects_grid(dammy_pfl):
    """
    Checks that function PhaseFieldLattice::is_occupied()
    works with the continuous logic.
    """
    lat = dammy_pfl
    assert lat.is_occupied(1, 1, 0) is False
    lat.grid[1, 1, 0] = 1
    assert lat.is_occupied(1, 1, 0) is True

def test_update_occupied_and_history_sets_grid_and_occupied(dammy_pfl):
    """
    Checks that when occupying a new cell of PhaseFieldLattice all the 
    objects are updated correctly.
    """
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
    """
    Checks that when setting a new nucleation seed of PhaseFieldLattice all the 
    objects are updated correctly.
    """
    lat = dammy_pfl
    z = 0

    lat.set_nucleation_seed(x=3, y=3, z=z, radius=2.0, width=0.5, phi_in=1.0, phi_out=0.0)

    assert (3, 3, z) in lat._seeds
    assert lat.phi[:, :, z].max() > 0.0

    assert np.any(lat.grid[:, :, z] == 1)
    assert len(lat.occupied) > 0

    occ = list(lat.occupied)
    x0, y0, z0 = occ[0]
    assert lat.group_id[x0, y0, z0] == 1

def test_nearest_seed_id_prefers_closest(dammy_pfl):
    """
    Test if the id logic also works in PhaseFieldLattice.
    """
    lat = dammy_pfl

    lat.set_nucleation_seed(1, 1, 0, radius=1.0, width=0.5)
    lat.set_nucleation_seed(5, 5, 0, radius=1.0, width=0.5)

    lat.phi[5, 5, 0] = 1.0
    lat.update_occupied_and_history(epoch=1)

    assert (5, 5, 0) in lat.occupied
    assert lat.group_id[5, 5, 0] == 2