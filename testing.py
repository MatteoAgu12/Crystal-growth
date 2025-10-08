import numpy as np
from Lattice import Lattice
import EDEN_simulation as EDEN
import DLA_simulation as DLA
import GUI as GUI

# === Lattice class ========================================================
def test_lattice_class():
    """
    This test function tests the attributes of the 'Lattice' class, including the '__init__()' function.
    
    A lattice with shape (3, 5, 7) is generated.
    Then the function 'is_point_inside()' is tested, using both point expected to be inside the lattice and not.
    The internal point (2, 2, 2) is then set to occupied using the function 'occupy()', and the function 'is_occupied()' is tested.
    To test the functions 'get_neighbors()' and 'get_active_border()', the output of these function is compared to the expected output.
    Finally, the function 'set_nucleation_seed()' is tested to assert if it correctly modify the occupation of the selected point and the attribute 'Lattice.initial_seeds'.
    """
    LATTICE = Lattice(3, 5, 7)
    assert LATTICE.shape == (3, 5, 7)
    
    assert LATTICE.is_point_inside(1, 1, 1)
    assert not LATTICE.is_point_inside(10, 10, 10)
    assert not LATTICE.is_point_inside(3, 5, 7)
    assert LATTICE.is_point_inside(0, 0, 0)
    assert LATTICE.is_point_inside(2, 4, 6)
    
    LATTICE.occupy(2, 2, 2, 0)
    assert LATTICE.is_occupied(2, 2, 2)
    assert not LATTICE.is_occupied(2, 2, 1)
    
    neighbors = LATTICE.get_neighbors(2 ,2, 2)
    expected_neighbors = [[1, 2, 2], [2, 1, 2], [2, 3, 2], [2, 2, 1], [2, 2, 3]] # point [3, 2, 2] is not inside this lattice!
    assert len(neighbors) == 5
    for expected_neighbor in expected_neighbors:
        assert expected_neighbor in neighbors
    
    active_border = LATTICE.get_active_border()
    expected_border = [[1, 2, 2], [2, 1, 2], [2, 3, 2], [2, 2, 1], [2, 2, 3]] # point [3, 2, 2] is not inside this lattice!
    assert len(active_border) == 5
    for expected_border_point in expected_border:
        assert expected_border_point in active_border
    
    assert LATTICE.initial_seeds == []
    LATTICE.set_nucleation_seed(0, 0, 0)
    assert np.array_equal(LATTICE.get_nucleation_seeds(), np.array([[0, 0, 0]]))
    assert LATTICE.is_occupied(0, 0, 0)
    assert len(LATTICE.initial_seeds) == 1
    assert (0, 0, 0) in LATTICE.initial_seeds
    assert not (2, 2, 2) in LATTICE.initial_seeds

def test_lattice_get_crystal_bounding_box_method():
    """
    Function to test the method 'Lattice.get_crystal_bounding_box()'.
    
    The bounding box returned by the function is compered with the expected one for different configurations of occupied sites and for different padding values.
    """
    lattice = Lattice(5, 5, 5)
    lattice.set_nucleation_seed(2, 2, 2)
    
    assert lattice.get_crystal_bounding_box() == ((2,3), (2,3), (2,3))
    assert lattice.get_crystal_bounding_box(padding=1) == ((1, 4), (1, 4), (1, 4))
    
    lattice.set_nucleation_seed(1, 1, 1)
    
    assert lattice.get_crystal_bounding_box() == ((1, 3), (1, 3), (1, 3))

def test_lattice_history_member_update():
    """
    This function tests the correct generation of the 'history' member and its update using the 'occupy()' method.
    A 1D lattice of lenght 6 is generated, and the first five cells are occupied in five consecutive steps.
    The function then checks the corresponding value in the history.
    Finally, it checks that the last cell, which is empty, has an associeted value of -1 in the history array.
    """
    LATTICE = Lattice(6, 1, 1)
    for i in range(5): LATTICE.occupy(i, 0, 0, epoch=i+1)
    
    assert LATTICE.history.size == 6
    for i in range(5): assert LATTICE.history[i, 0, 0] == i+1
    assert LATTICE.history[5, 0, 0] == -1

# === EDEN simulation section ==============================================    
def test_choose_random_border_site_function():
    """
    This test function tests the 'choose_random_border_site()' function.
    
    A dummy active border is defined, together with an empty dummy border, which is an empty list.
    First the function is tested on the empty one, to see if it returns None.
    Then, the function is called 10 times on the dummy border, and it is asserted that the output is one of the element of teh border itself.
    """
    dummy_active_border = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]])
    empty_dummy_border = []
    
    assert EDEN.choose_random_border_site(empty_dummy_border) is None
    for _ in range(10):
        assert EDEN.choose_random_border_site(dummy_active_border) in dummy_active_border
        
def test_EDEN_simulation_function():
    """
    This function tests the 'EDEN_simulation()' function.
    
    To do the test, three lattices are defined: one with no nucleation sites, one fully occupied and one normal.
    The output code of the function is tested for all these three situations and compared with the expected one.
    """
    N_reps = 10
    lattice_with_no_initial_seeds = Lattice(10, 10, 10)
    
    lattice_with_initial_seed = Lattice(10, 10, 10)
    lattice_with_initial_seed.set_nucleation_seed(5, 5, 5) 
    
    fully_occupied_lattice = Lattice(1, 1, 1)
    fully_occupied_lattice.set_nucleation_seed(0, 0, 0)
    
    assert EDEN.EDEN_simulation(lattice_with_initial_seed, N_reps) == 0
    assert EDEN.EDEN_simulation(lattice_with_no_initial_seeds, N_reps) == 1
    assert EDEN.EDEN_simulation(fully_occupied_lattice, N_reps) == 2

# === DLA simulation section ===============================================
def test_generate_random_point_on_box_function():
    """
    This function tests the 'generate_random_point_on_box()' function.
    
    A random point is generated on the selected box using the tested function.
    Since it is a random function, it is tested if the point is actually generated on the surface of the box for 1000 different points.
    """
    box = ([1, 10], [36, 98], [78, 110])
    for _ in range(1000):
        generated_point = DLA.generate_random_point_on_box(box)
        assert ((1 <= generated_point[0] <= 10) and 
                (36 <= generated_point[1] <= 98) and 
                (78 <= generated_point[2] <= 110)) and ((generated_point[0] in box[0]) or 
                                                        (generated_point[1] in box[1]) or 
                                                        (generated_point[2] in box[2]))

def test_particle_random_walk_function():
    lattice = Lattice(10, 10, 10)
    lattice.set_nucleation_seed(5, 5, 5)
    repetitions = 100
    
    for rep in range(repetitions):
        lattice_bbox = lattice.get_crystal_bounding_box()
        generation_pos = DLA.generate_random_point_on_box(lattice_bbox)
        _, _ = DLA.particle_random_walk(lattice, generation_pos)

# === SURFACE simulation section ===========================================
def test_active_surface_seeds_preserved():
    L = Lattice(8,5,8)
    for x in range(8):
        for z in range(8):
            L.set_nucleation_seed(x,0,z)
    DLA.DLA_simulation(L, N_particles=500, generation_padding=1, outer_limit_padding=3, three_dim=True, verbose=False)
    assert np.all(L.history[:,0,:] == 0), "Some cells have been rewritten!"
            

# === GUI section ==========================================================        
def test_get_visible_voxels_binary_mask_function():
    """
    This function tests the 'get_visible_voxels_binary_mask()' function.
    
    A 3x3x3 lattice is generated, and its central cell (1, 1, 1) is set to occupied, together with all its neighbors.
    The mask is generated with the tested function, and the occupation of the output is tested.
    The test is performed by checking the central cell, that now is expected to be False, and the one of all others occupied, expected to be True.
    """
    lattice = Lattice(3, 3, 3)
    lattice.occupy(1, 1, 1, 0)
    lattice.occupy(1, 1, 0, 0)
    lattice.occupy(1, 1, 2, 0)
    lattice.occupy(1, 0, 1, 0)
    lattice.occupy(1, 2, 1, 0)
    lattice.occupy(0, 1, 1, 0)
    lattice.occupy(2, 1, 1, 0)
    visible_voxels = GUI.get_visible_voxels_binary_mask(lattice)
    
    assert np.sum(lattice.grid) == 7
    assert np.sum(visible_voxels) == 6
    assert not visible_voxels[1,1,1]
    
    
    
if __name__ == '__main__':
    test_lattice_class()
    test_lattice_get_crystal_bounding_box_method()
    test_lattice_history_member_update()
    
    test_choose_random_border_site_function()
    test_EDEN_simulation_function()
    
    test_generate_random_point_on_box_function()
    test_particle_random_walk_function()
    
    test_active_surface_seeds_preserved()
    
    test_active_surface_seeds_preserved()