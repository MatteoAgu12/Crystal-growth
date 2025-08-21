import numpy as np
from Lattice import Lattice

# TODO: ancora da testare, mettere in 2D, ecc...
def compute_radial_density(lattice: Lattice, max_radius: int = None, dr: float = 1.0) -> tuple:
    grid = lattice.grid
    shape = lattice.shape
    center = lattice.get_nucleation_seeds()[0]
    
    z, y, x = np.indices(shape)
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)
    
    if max_radius is None:
        max_radius = int(np.max(r))
        
    bins = np.arange(0, max_radius + dr, dr)
    bin_centers = (bins[:1] + bins[1:]) / 2
    densities = []
    
    for i in range(len(bins)-1):
        r_min, r_max = bins[i], bins[i+1]
        shell_mask = (r_min <= r < r_max)
        shell_occupied = grid[shell_mask]
        
        n_total = shell_mask.sum()
        n_occupied = shell_occupied.sum()
        
        density = n_occupied / n_total if n_total > 0 else 0
        densities.append(density)
    
    return (bin_centers, np.array(densities))