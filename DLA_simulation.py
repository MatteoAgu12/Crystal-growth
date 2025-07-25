import numpy as np
from Lattice import Lattice

def generate_random_point_on_sphere(radius: float, center: list = [0, 0, 0]) -> np.array:
    """
    This function randomly generates a point on a sphere of a selected radius and center.
    Each coordinate of the point given as output is casted to int, in order to match with the Lattice object functionalities.

    Args:
        radius (float): radius of the sphere.
        center (list, optional): coordinates of the center point of the sphere. Defaults to [0, 0, 0].

    Raises:
        Valuerror: if the input parameter 'radius' is less then or equal to zero, the program stops.
        ValueError: if the input parameter 'center' is not a list of lenght == 3, the program stops.

    Returns:
        np.array: _description_
    """
    if radius <= 0:
        raise ValueError(f"ERROR: in function 'generate_random_point_on_sphere' the radius input parameter \
            must be a number bigger than zero. \
            \nThe value {radius} has been inserted. Aborted")
    if len(center) != 3:
        raise ValueError(f"ERROR: in function 'generate_random_point_on_sphere' the center input parameter \
            must be the set of coordinates of the center of the sphere. \
            \nThe value {center} has been inserted. Aborted")
        
    phi = np.random.uniform(0, 1) * 2 * np.pi
    theta = np.random.uniform(0, 1) * np.pi
    
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    cos_f, sin_f = np.cos(phi), np.sin(phi)
    
    return np.array([int(center[0] + radius * cos_f * sin_t), 
                     int(center[1] + radius * sin_t, sin_f), 
                     int(center[2] + radius * cos_t)])
    
def particle_random_walk(lattice: Lattice, initial_coordinate: np.array, outer_allowed_bounding_box: tuple) -> tuple:
    """
    This function creates a particle in position 'initial_coordinate' and performes a random walk.
    If the particles arrives in a site with an occupied neighbor, it stops and becomes part of the crystal.
    If the particle exits from the bounding box 'outer_allowed_bounding_box', its position is set to the initial one and the random walk restarts.

    Args:
        lattice (Lattice): custom Lattice object.
        initial_coordinate (np.array): coordinates of the spawn point of the new particle.
        outer_bounding_box (tuple): bounding box outside which the particle can't go. If it happens, the random walk restarts.

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
                outer_allowed_bounding_box[2][0] <= position[2] <= outer_allowed_bounding_box[2][1]):
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
            
    

if __name__ == '__main__':
    out = np.random.randint(-1, 2, 10)
    print(out)