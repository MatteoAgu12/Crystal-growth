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
    position = initial_coordinate.copy()
    continue_walk = True
    number_of_restarts = 0
    total_steps = 0
    
    while continue_walk:
        total_steps += 1
        move_along = np.random.randint(0, 3)
        direction = np.random.choice([-1, 1])
        position[move_along] += direction
        print(position)
        
        (xmin, xmax), (ymin, ymax), (zmin, zmax) = outer_allowed_bounding_box
        if not (xmin <= position[0] <= xmax and ymin <= position[1] <= ymax and zmin <= position[2] <= zmax) or total_steps > max_steps:
            print("RESTARTED!!!!!!")
            position = initial_coordinate.copy() 
            number_of_restarts += 1 
            total_steps = 0        
        
        else:
            neighbors = lattice.get_neighbors(position[0], position[1], position[2])        
            for neighbor in neighbors:
                if lattice.is_occupied(neighbor[0], neighbor[1], neighbor[2]): 
                    lattice.occupy(int(position[0]), int(position[1]), int(position[2]))
                    continue_walk = False
                    break
            
    return (total_steps, number_of_restarts)
            
def DLA_simulation(lattice: Lattice, N_particles: int, generation_padding: int, outer_limit_padding: int) -> tuple:
    if N_particles <= 0:
        raise ValueError(f"ERROR: in function 'DLA_simulation()' the number of generated particles must be an integer bigger than zero. \
            You inserted {N_particles}. Aborted")
        
    if outer_limit_padding <= generation_padding:
        raise ValueError(f"ERROR: in function 'DLA_simulation()' the generation padding must be smaller than the outer limit one. \
            You inserted {generation_padding} and {outer_limit_padding} respectively. Aborted.")    
    
    steps = np.zeros(N_particles)
    restarts = np.zeros(N_particles)
        
    for n in range(N_particles):
        generation_box = lattice.get_crystal_bounding_box(padding=generation_padding)
        outer_limit_box = lattice.get_crystal_bounding_box(padding=outer_limit_padding)        
        starting_point = generate_random_point_on_box(generation_box)
        
        print(f"Starting point = {starting_point}")
        
        n_step, n_restart = particle_random_walk(lattice, starting_point, outer_limit_box)
        steps[n] = n_step
        restarts[n] = n_restart
    
    step_mean, step_std = np.mean(steps), np.std(steps)
    restart_mean, restart_std = np.mean(restarts), np.std(restarts)
    
    return (step_mean, step_std, restart_mean, restart_std)

