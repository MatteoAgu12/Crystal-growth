import numpy as np
from Lattice import Lattice

def generate_random_point_on_box(bounding_box: tuple) -> np.array:
    """
    This function randomly generates a point on the surface of a box.
    It randomly selects one face of the box, and then it generates a random point constrained on that face

    Args:
        bounding_box (tuple): bounding box on which generate the point.

    Raises:
        ValueError: if the input parameter 'bounding_box' is not an object of lenght == 3, the program stops.

    Returns:
        (np.array): the coordinates of the randomly generated point
    """
    if len(bounding_box) != 3:
        raise ValueError(f"ERROR: in function 'generate_random_point_on_sphere' the center input parameter \
            must be the set of coordinates of the center of the sphere. \
            \nThe value {bounding_box} has been inserted. Aborted")
        
    axis = np.random.randint(0, 3)
    face = np.random.randint(0, 2)
    point = np.zeros(3)
    
    for direction in range(3):
        if direction == axis:
            point[direction] = int(bounding_box[direction][face])
        else:
            point[direction] = int(np.random.randint(bounding_box[direction][0], bounding_box[direction][1]+1))
           
    return point
    
def particle_random_walk(lattice: Lattice, initial_coordinate: np.array, outer_allowed_bounding_box: tuple, max_steps: int = 1000) -> tuple:
    """
    This function creates a particle in position 'initial_coordinate' and performes a random walk.
    If the particles arrives in a site with an occupied neighbor, it stops and becomes part of the crystal.
    If the particle exits from the bounding box 'outer_allowed_bounding_box', its position is set to the initial one and the random walk restarts.

    Args:
        lattice (Lattice): custom Lattice object.
        initial_coordinate (np.array): coordinates of the spawn point of the new particle.
        outer_bounding_box (tuple): bounding box outside which the particle can't go. If it happens, the random walk restarts.
        max_steps (int, optional): maximum number of step performed before restarting the walk.

    Returns:
        (tuple): cuple of int representing the number of stpes needed to reach the crystal (in a cycle) and how many times the walk has restarted.
    """
    position = initial_coordinate
    continue_walk = True
    number_of_restarts = 0
    total_steps = 0
    
    while continue_walk:
        total_steps += 1
        position += np.random.randint(-1, 2, 3)
        
        if not (outer_allowed_bounding_box[0][0] <= position[0] <= outer_allowed_bounding_box[0][1] and 
                outer_allowed_bounding_box[1][0] <= position[1] <= outer_allowed_bounding_box[1][1] and
                outer_allowed_bounding_box[2][0] <= position[2] <= outer_allowed_bounding_box[2][1]) or total_steps > max_steps:
            position = initial_coordinate 
            number_of_restarts += 1 
            total_steps = 0        
        
        neighbors = lattice.get_neighbors(position[0], position[1], position[2])        
        for neighbor in neighbors:
            if lattice.is_occupied(neighbor[0], neighbor[1], neighbor[2]): 
                lattice.occupy(position[0], position[1], position[2])
                continue_walk = False
                break
            
    return (total_steps, number_of_restarts)
            
def DLA_simulation(lattice: Lattice, N_particles):    
    pass

if __name__ == '__main__':
    out = np.random.randint(-1, 2, 10)
    print(out)