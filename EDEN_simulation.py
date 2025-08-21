import numpy as np
from typing import Union
from Lattice import Lattice
import GUI

def choose_random_border_site(active_border: np.array) -> Union[np.array, None]:
    """This function randomly selects a site from the active border.

    Args:
        active_border (np.array): active border of the crystal, obtained via Lattoce.get_active_border().

    Returns:
        (np.array): coordinates of the randomly selected site of the active border.
    """
    if len(active_border) == 0:
        return None
    
    return np.array(active_border[np.random.randint(0, len(active_border))])

def EDEN_simulation(lattice: Lattice, N_reps: int, three_dim : bool = True) -> int:
    """
    This function performs a crystal growth EDEN simulation (saturated enviroment).

    Args:
        lattice (Lattice): custom lattice object
        N_reps (int): number of rpetitions to run (maximum number of new cells to add)
        three_dim (bool, optional): decides if the crystal is two or three dimentional. Defaults to True.

    Raises:
        ValueError: if the input parameter N_reps is less than or equal to zero, the function raises an error.
                    This is because a negative number of repetitions is nonsense.
        ValueError: if you select a 2D simulation and the initial nucleation seeds don't have the same z-coord, the function raises error.

    Returns:
        int: a numeric code representing how the simulation ends.
                * 0: simulation correctly executed N_reps simulation steps.
                * 1: in the lattice there are no initial nucleation seeds, therefore it is impossible to perform a simulation.
                * 2: all the cells of the lattice are already occupied, so there is no space left.
                * 3: other reasons
    """
    if N_reps <= 0:
        raise ValueError(f"ERROR: in function 'EDEN_simulation' the parameter N_reps must be an integer bigger than zero, you inserted {N_reps}")
    
    if not three_dim:
        seeds_on_same_xy_plane = True
        seeds = lattice.get_nucleation_seeds()
        for seed in seeds:
            seeds_on_same_xy_plane = (seeds[0][2] == seed[2])
            if not seeds_on_same_xy_plane: break
            
        if not seeds_on_same_xy_plane:
            raise ValueError("ERROR: in function 'DLA_simulation()', in a 2D simulation the nucleation seeds must have the same z-coord.")
    
    z_coord = lattice.get_active_border()[0][2] if not three_dim else None
    for n in range(N_reps):
        active_border = lattice.get_active_border()
        planar_border = [] if not three_dim else None
        if not three_dim:
            for site in active_border:
                if site[2] == z_coord: planar_border.append(site)
            
        new_cell = choose_random_border_site(active_border) if three_dim else choose_random_border_site(planar_border)
        
        if new_cell is None:
            if len(lattice.initial_seeds) == 0: return 1
            elif len(active_border) == 0: return 2
            else: return 3
        
        x, y, z = new_cell
        lattice.occupy(x, y, z, epoch=n+1)
        
        print(f"Procedure completed for particle {n+1}")
        
    return 0
            
