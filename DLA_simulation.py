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
    
def particle_random_walk(lattice: Lattice, initial_coordinate: np.array, max_steps: int = 1000, verbose: bool = False) -> int:
    # TODO: add a way to restart the random walk if the particle goes too far from the crystal
    position = initial_coordinate
    
    for i in range(max_steps):
        position += np.random.randint(-1, 2, 3)
        neighbors = lattice.get_neighbors(position[0], position[1], position[2])
        
        for neighbor in neighbors:
            if lattice.is_occupied(neighbor[0], neighbor[1], neighbor[2]): 
                lattice.occupy(position[0], position[1], position[2])
                
                if verbose:
                    print(f"Particle attached in {i} steps.")
                
                return 0
            
    return 1
            
    

if __name__ == '__main__':
    out = np.random.randint(-1, 2, 10)
    print(out)