import numpy as np
from Lattice import Lattice

def compute_fractal_dimention(lattice: Lattice, min_box_size : int = 2, max_box_size: int = None, num_scales: int = 10, 
                           three_dim: bool = True) -> tuple:
    """
    Computes the fractal (Hausdorff) dimention of a single crystal using the box-counting method.
    
    Args:
        lattice (Lattice): custom Lattice object after the growth simulation.
        min_box_size (int): min box size (in pixel).
        max_box_size (int): max box size. Default None, in this case set to min(grid.shape)/2
        num_scales (int): number of log scale to use. Default 10.
        three_dim (bool): if the crystal is 2D or 3D. Dafault to True (3D).
        
    Returns:
        (tuple): tuple in the form (D = fractal dimention, sizes = array of log scales used, counts = box with overlap per scale).
    """
    grid = lattice.grid
    if max_box_size is None:
        max_box_size = min(grid.shape) // 2 if three_dim else min(grid.shape[:2]) // 2
    
    sizes = np.logspace(np.log10(min_box_size), np.log10(max_box_size), num=num_scales, dtype=int)
    sizes = np.unique(sizes)
    
    counts = []
    for size in sizes:
        count = 0
        for i in range(0, grid.shape[0], size):
            for j in range(0, grid.shape[1], size):
                
                if three_dim:
                    for k in range(0, grid.shape[2], size):
                        if np.any(grid[i:i+size, j:j+size, k:k+size]):
                            count += 1
                else:
                    if np.any(grid[i:i+size, j:j+size]):
                        count += 1
                        
        counts.append(count)
    
    coeffs = np.polyfit(np.log(1/sizes), np.log(counts), 1)
    D = coeffs[0]
    
    return (D, sizes, counts)
