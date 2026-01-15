import numpy as np
from abc import ABC

class BaseLattice(ABC):
    def __init__(self, nx: int, ny: int, nz: int, verbose: bool):
        if nx < 0 or ny < 0 or nz < 0:
            raise ValueError("Lattice dimensions must be >= 0")

        self.shape = (nx, ny, nz)

        self.history   = np.ones(self.shape, dtype=np.int64) * (-1)
        self.group_id  = np.zeros(self.shape, dtype=np.uint16)
        self.group_counter = 0
        self.initial_seeds = []
        self.verbose = verbose

    def __str__(self):
        return f"""
        === Lattice Object ======================================== 
         * Shape:               {self.shape}
         * Nucleation seeds:    {self.initial_seeds}
         * Verbose:             {self.verbose}
        ============================================================
         """

    
    def is_point_inside(self, x: int, y: int, z: int) -> bool:
        """
        Returns if a point (x,y,z) is or not inside the lattice.

        Args:
            x (int): x coordinate of the point to verify
            y (int): y coordinate of the point to verify
            z (int): z coordinate of the point to verify
            
        Returns:
            (bool): returns if the point is inside the lattice or not
        """
        nx, ny, nz = self.shape
        return (0 <= x < nx) and (0 <= y < ny) and (0 <= z < nz)
        
    def get_neighbors(self, x: int, y: int, z: int) -> np.array:
        """
        Function that returns the coordinates of the six cells around the selected one (only adjecent cells, not diagonally).
        
        Args:
            x (int): x coordinate of the point to check the neighbors
            y (int): y coordinate of the point to check the neighbors
            z (int): z coordinate of the point to check the neighbors
            
        Returns:
            (np.array): array containing the coordinates of the six cell neighbors
        """
        directions = [
            (1, 0, 0), (-1, 0, 0),  # x-axis
            (0, 1, 0), (0, -1, 0),  # y-axis
            (0, 0, 1), (0, 0, -1),  # z-axis
        ]
        neighbors = []
        for dx, dy, dz in directions:
            nx, ny, nz = x + dx, y + dy, z + dz
            
            if self.is_point_inside(nx, ny, nz):
                neighbors.append([int(nx), int(ny), int(nz)])
                
        return np.array(neighbors)
