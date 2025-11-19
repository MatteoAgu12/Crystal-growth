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
    
def particle_random_walk(lattice: Lattice, initial_coordinate: np.array, outer_allowed_bounding_box: tuple, epoch: int,
                         max_steps: int = 100, three_dim : bool = True) -> tuple:
    """
    This function creates a particle in position 'initial_coordinate' and performes a random walk.
    If the particles arrives in a site with an occupied neighbor, it stops and becomes part of the crystal.
    If the particle exits from the bounding box 'outer_allowed_bounding_box', its position is set to the initial one and the random walk restarts.

    Args:
        lattice (Lattice): custom Lattice object.
        initial_coordinate (np.array): coordinates of the spawn point of the new particle.
        outer_bounding_box (tuple): bounding box outside which the particle can't go. If it happens, the random walk restarts.
        epoch (int): current epoch number.
        max_steps (int, optional): maximum number of step performed before restarting the walk.
        three_dim (bool, optional): decides if the crystal is two or three dimentional. Defaults to True.

    Returns:
        (tuple): cuple of int representing the number of stpes needed to reach the crystal (in a cycle) and how many times the walk has restarted.
    """
    position = initial_coordinate.copy()
    continue_walk = True
    number_of_restarts = 0
    total_steps = 0
    
    while continue_walk:
        total_steps += 1
        move_along = np.random.randint(0, 3) if three_dim else np.random.randint(0, 2)
        direction = np.random.choice([-1, 1])
        position[move_along] += direction
        
        (xmin, xmax), (ymin, ymax), (zmin, zmax) = outer_allowed_bounding_box
        if not (xmin <= position[0] <= xmax and ymin <= position[1] <= ymax and zmin <= position[2] <= zmax) or total_steps > max_steps:
            position = initial_coordinate.copy() 
            number_of_restarts += 1 
            total_steps = 0        
        
        else:
            neighbors = lattice.get_neighbors(position[0], position[1], position[2])        
            for neighbor in neighbors:
                if lattice.is_occupied(neighbor[0], neighbor[1], neighbor[2]): 
                    lattice.occupy(int(position[0]), int(position[1]), int(position[2]), epoch=epoch)
                    continue_walk = False
                    break
            
    return (total_steps, number_of_restarts)
            
def DLA_simulation(lattice: Lattice, N_particles: int, generation_padding: int, outer_limit_padding: int, 
                   three_dim : bool = True,
                   verbose: bool = True) -> tuple:
    """
    This function performs a crystal growth DLA simulation (diffusion limited enviroment).

    Args:
        lattice (Lattice): custom lattice object.
        N_particles (int): number of particle to deposit.
        generation_padding (int): padding to the crystal bounding box. 
                                  It is the nearest limit on which particles are generated at the beginning of the random walk.
        outer_limit_padding (int): padding to the crystal bounding box.
                                   During walk, if a particle exits from this box, the random walk restarts.
        three_dim (bool, optional): decides if the crystal is two or three dimentional. Defaults to True.
        verbose (bool, optional): if true prints additional info during the simulation. Defaults to True.

    Raises:
        ValueError: if the input N_particles is less than or equal to zero, the function raises an error.
                    This is because a negative number of particles is nonsense.
        ValueError: if the nearest padding is higher than the farest, the function raises an error.
        ValueError: if you select a 2D simulation and the initial nucleation seeds don't have the same z-coord, the function raises error.

    Returns:
        tuple: statistics about the simulation, reguarding the random wlak, in the form (step_mean, step_std, restart_mean, restart_std).
    """
    if N_particles <= 0:
        raise ValueError(f"ERROR: in function 'DLA_simulation()' the number of generated particles must be an integer bigger than zero. \
            You inserted {N_particles}. Aborted")
        
    if outer_limit_padding <= generation_padding:
        raise ValueError(f"ERROR: in function 'DLA_simulation()' the generation padding must be smaller than the outer limit one. \
            You inserted {generation_padding} and {outer_limit_padding} respectively. Aborted.")
        
    if not three_dim:
        seeds_on_same_xy_plane = True
        seeds = lattice.get_nucleation_seeds()
        for seed in seeds:
            seeds_on_same_xy_plane = (seeds[0][2] == seed[2])
            if not seeds_on_same_xy_plane: break
            
        if not seeds_on_same_xy_plane:
            raise ValueError("ERROR: in function 'DLA_simulation()', in a 2D simulation the nucleation seeds must have the same z-coord.")
    
    steps = np.zeros(N_particles)
    restarts = np.zeros(N_particles)
    z_coord = lattice.get_nucleation_seeds()[0][2] if not three_dim else None
        
    update_step = int(N_particles / 10)
    completing_percentage = 0
    for n in range(N_particles):        
        if not verbose and n == completing_percentage * update_step:
            if completing_percentage != 0:
                print(f"=== Simulation completed at {int(completing_percentage*10)}% ===")
            completing_percentage += 1
        generation_box = lattice.get_crystal_bounding_box(padding=generation_padding)
        outer_limit_box = lattice.get_crystal_bounding_box(padding=outer_limit_padding)
        
        if not three_dim:
            generation_box[2] = (z_coord, z_coord)
            outer_limit_box[2] = (z_coord, z_coord)
                
        starting_point = generate_random_point_on_box(generation_box)
        
        n_step, n_restart = particle_random_walk(lattice, starting_point, outer_limit_box, epoch=n+1, three_dim=three_dim)
        steps[n] = n_step
        restarts[n] = n_restart
        
        if verbose: print(f"Procedure completed for particle {n+1}")
    
    step_mean, step_std = np.mean(steps), np.std(steps)
    restart_mean, restart_std = np.mean(restarts), np.std(restarts)
    
    return (step_mean, step_std, restart_mean, restart_std)

