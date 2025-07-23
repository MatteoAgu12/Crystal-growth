from Lattice import Lattice

def test_lattice_class():
    """
    This test function tests the attributes of the Lattice class, including the __init__() function.
    
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
    
    LATTICE.occupy(2, 2, 2)
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
    assert LATTICE.is_occupied(0, 0, 0)
    assert len(LATTICE.initial_seeds) == 1
    assert (0, 0, 0) in LATTICE.initial_seeds
    assert not (2, 2, 2) in LATTICE.initial_seeds
