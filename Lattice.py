import numpy as np
from typing import Union

class Lattice:
    """
    """
    def __init__(self, number_of_cells_x: int, number_of_cells_y: int, number_of_cells_z: int):
        """
        Args:
            number_of_cells_x (int): number of cells in the lattice along the x direction
            number_of_cells_y (int): number of cells in the lattice along the y direction
            number_of_cells_z (int): number of cells in the lattice along the z direction
        """
        if number_of_cells_x < 0 or number_of_cells_y < 0 or number_of_cells_z < 0:
            raise ValueError('ERROR: the size of the lattice must be an integer bigger or equal to zero!')
            
        self.shape = (number_of_cells_x, number_of_cells_y, number_of_cells_z)
        self.grid = np.zeros(self.shape, dtype=np.uint8)    # 0 = empty; 1 = occupied
        self.history = np.ones(self.shape, dtype=np.int64) * (-1)
        self.initial_seeds = []
        self.occupied = set()

    def __str__(self):
        return f"Lattice has shape: {self.shape} \
            \nNucleation seeds: {self.initial_seeds} \
            \nNumber of occupied sites: {np.sum(self.grid)}"

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

    def occupy(self, x: int, y: int, z: int, epoch: int) -> None:
        """
        Set the status of the cell at coordinates (x,y,z) to occupied.
        It also keeps track of the epoch at which the new cell has been occupied.
        
        Args:
            x (int): x coordinate of the point to occupy
            y (int): y coordinate of the point to occupy
            z (int): z coordinate of the point to occupy
            epoch (int): current epoch in the simulation
            
        Raises:
            ValueError: if the epoch number is negative the function raises error.
        """
        if epoch < 0:
            raise ValueError(f"ERROR: in function 'Lattice::occupy()' the epoch number must be a an integer bigger or equal to zero.")
        
        if not self.is_point_inside(x, y, z):
            return
        
        if (x, y, z) in self.occupied:
            return
        
        self.grid[x, y, z] = 1
        self.history[x, y, z] = int(epoch)
        self.occupied.add((int(x), int(y), int(z)))
            
    def is_occupied(self, x: int, y: int, z: int) -> bool:
        """
        Function that check if a point at coordinates (x,y,z) i occupied or not.

        Args:
            x (int): x coord of the pixel
            y (int): y coord of the pixel
            z (int): z coord of the pixel
            
        Returns:
            (bool): if that pixel is occupied or not
        """
        return (int(x), int(y), int(z)) in self.occupied

    def set_nucleation_seed(self, x: int, y: int, z: int) -> None:
        """
        Function that initialize one nucleation seed.
        The function does nothing if the selected point is outside the lattice or if it's already a nucleation seed.

        Args:
            x (int): x coordinate of the nucleation seed
            y (int): y coordinate of the nucleation seed
            z (int): z coordinate of the nucleation seed
        """
        if self.is_point_inside(x, y, z) and (x, y, z) not in self.initial_seeds:
            self.occupy(x, y, z, epoch=0)
            self.initial_seeds.append((x, y, z))
    
    def get_nucleation_seeds(self) -> np.array:
        """
        This function returns the initial nucleation seeds.

        Returns:
            np.array: array containing all the initial nucleation seeds.
        """
        return np.array(self.initial_seeds)
    
    def get_active_border(self):
        """
        Function that compute the active border of teh crystal.
        The active border is the set of all empty cells having at least one occupied cell as neighbor.

        Returnss:
            (np.array): array containing the points that form the active border
        """
        active_set = set()
                
        for (x, y, z) in self.occupied:
            for (nx, ny, nz) in self.get_neighbors(x, y, z):
                if (int(nx), int(ny), int(nz)) not in self.occupied:
                    active_set.add((int(nx), int(ny), int(nz)))
                    
        if not active_set: return np.array([]).reshape((0, 3))
        return np.array(list(active_set), dtype=int)
 
    def get_crystal_bounding_box(self, padding: int = 0) -> Union[list, None]:
        """
        Function to compute the bounding box of the occupied region (smallest parallelogram that contains it).

        Args:
            padding (int, optional): optional enlargement (in each direction) of the box. Defaults to 0.

        Returns:
            (Union[list, None]): tuple containing the information (coord_min, coord_max) for each coordinate.
        """
        if not self.occupied:
            return None
        
        occupied_coords = np.array(list(self.occupied), dtype=int)
        mins = occupied_coords.min(axis=0) - padding
        maxs = occupied_coords.max(axis=0) + 1 + padding
        
        # I never want to exit from my lattice grid
        mins = np.clip(mins, 0, np.array(self.shape) - 1)
        maxs = np.clip(maxs, 0, np.array(self.shape) - 1)
        
        return list(zip(mins, maxs))
        



