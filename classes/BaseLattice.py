import numpy as np
from abc import ABC, abstractmethod

class BaseLattice(ABC):
    """
    This class represents the base lattice structure for crystal simulations. 
    It defines the basic properties and methods that any specific lattice implementation should have. 
    The lattice is represented as a 3D grid, where each cell can be occupied or unoccupied.
    It keeps track of the history of occupation and group IDs for nucleation processes.
    """
    def __init__(self, nx: int, ny: int, nz: int, verbose: bool):
        """
        Args:
            nx (int): number of cells in the lattice along the x direction
            ny (int): number of cells in the lattice along the y direction
            nz (int): number of cells in the lattice along the z direction
            verbose (bool): if True, the lattice will print debug information during initialization and occupation of cells.

        Raises:
            ValueError: if the size of the lattice is negative, the function raises error.
        """
        if nx < 0 or ny < 0 or nz < 0:
            raise ValueError("KineticLattice dimensions must be >= 0")

        self.dx = 0.03
        self.shape = (nx, ny, nz)

        self.history   = np.ones(self.shape, dtype=np.int64) * (-1)
        self.group_id  = np.zeros(self.shape, dtype=np.uint16)
        self.occupied = set()
        self.group_counter = 0
        self.initial_seeds = []
        self.verbose = verbose

    def __str__(self):
        return f"""
        === KineticLattice Object ======================================== 
         * Shape:               {self.shape}
         * Nucleation seeds:    {self.initial_seeds}
         * Verbose:             {self.verbose}
        ==================================================================
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
        
    def get_neighbors(self, x: int, y: int, z: int) -> np.ndarray:
        """
        Function that returns the coordinates of the six cells around the selected one (only adjecent cells, not diagonally).
        
        Args:
            x (int): x coordinate of the point to check the neighbors
            y (int): y coordinate of the point to check the neighbors
            z (int): z coordinate of the point to check the neighbors
            
        Returns:
            (np.ndarray): array containing the coordinates of the six cell neighbors
        """
        directions = [
            (1, 0, 0), (-1, 0, 0),
            (0, 1, 0), (0, -1, 0),
            (0, 0, 1), (0, 0, -1),
        ]
        neighbors = []
        for dx, dy, dz in directions:
            nx, ny, nz = x + dx, y + dy, z + dz
            
            if self.is_point_inside(nx, ny, nz):
                neighbors.append([int(nx), int(ny), int(nz)])
                
        return np.array(neighbors)

    @abstractmethod
    def save_frame(self, epoch: int, three_dim: bool, frame_dir: str, frame_list: list) -> str:
        """
        Saves a frame of the current status into a directory.

        Args:
            epoch (int): epoch number
            three_dim (bool): if True, the lattice is three dimentional
            frame_dir (str): directory where to save the frame
            frame_list (list): list that collects all the frame saved

        Returns:
            str: the name of tyhe file containing the frame
        """
        pass