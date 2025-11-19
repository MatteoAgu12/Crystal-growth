import numpy as np
import matplotlib.pyplot as plt
from Lattice import Lattice

def compute_fractal_dimention(lattice: Lattice, min_box_size : int = 2, max_box_size: int = None, num_scales: int = 10, 
                           three_dim: bool = True) -> tuple:
    """
    Computes the fractal (Hausdorff) dimention of a single crystal using the box-counting method.
    
    Args:
        lattice (Lattice): custom Lattice object after the growth simulation.
        min_box_size (int, optional): min box size (in pixel).
        max_box_size (int, optional): max box size. Default None, in this case set to min(grid.shape)/2
        num_scales (int, optional): number of log scale to use. Default 10.
        three_dim (bool, optional): if the crystal is 2D or 3D. Dafault to True (3D).
        
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

def fractal_dimention_analysis(lattice: Lattice, output_dir: str,
                               min_box_size : int = 2, max_box_size: int = None, num_scales: int = 10, 
                               three_dim: bool = True, verbose: bool = True):
    """
    This function produces the plot needed to compute the Hausdorff dimention and save it in .png in the specified directory.

    Args:
        lattice (Lattice): custom Lattice object after the growth simulation.
        output_dir (str): directory in which save the produce plot.
        min_box_size (int, optional): min box size (in pixel).
        max_box_size (int, optional): max box size. Default None, in this case set to min(grid.shape)/2
        num_scales (int, optional): number of log scale to use. Default 10.
        three_dim (bool, optional): if the crystal is 2D or 3D. Dafault to True (3D).
        verbose (bool, optional): if True prints additional info during the analysis. Defaults to True.
    """
    
    D, sizes, counts = compute_fractal_dimention(lattice, min_box_size=min_box_size, max_box_size=max_box_size, num_scales=num_scales, 
                                                 three_dim=three_dim)
    
    print("\nAnalysis of the Hausdorff (fractal) dimention completed!")
    print(f"Computed fractal dimention: {D:.4f}")

    plt.figure()
    plt.title("Hausdorff dimention of the generated crystal")
    plt.plot(np.log(1/sizes), np.log(counts), "o-", label="data")
    plt.plot(np.log(1/sizes),
             np.polyval(np.polyfit(np.log(1/sizes), np.log(counts), 1), np.log(1/sizes)),
             "--", label=r"Best-fit (D $\approx$ " + f"{D:.4f})")
    plt.xlabel(r"-log$\epsilon$")
    plt.ylabel(r"log(N($\epsilon$))")
    plt.legend()
    
    filename = output_dir + "Hausdorff_dimention.png"
    plt.savefig(filename)
        
    print(f"\nImage of Hausdorff estimation analysis saved as {filename}.")

def distance_from_active_surface(lattice: Lattice, output_dir: str, N_epochs: int, verbose: bool = True):
    history = lattice.history
    time = np.linspace(1, N_epochs, N_epochs)
    distance = np.zeros(len(time))
    max_distance = 0
    
    for epoch in range(N_epochs):
        y_index = np.where(history == epoch+1)[1][0] if len(np.where(history == epoch+1)[1]) > 0 else None
        if y_index is not None:
            if y_index > max_distance:
                max_distance = y_index
                
        distance[epoch] = max_distance
            
    print("\nAnalysis of the distance of the farest cell from the active surface as function of time completed!")

    plt.figure()
    plt.title("Farest occupied cell VS time")
    plt.plot(time, distance, label="data", c='b')
    plt.xlabel("Epoch")
    plt.ylabel("Farest occupied distance")
    plt.legend()
    
    filename = output_dir + "Active_surface_distance.png"
    plt.savefig(filename)
        
    print(f"Plot of distance of the farest cell from the active surface as function of time saved as {filename}.")       
    