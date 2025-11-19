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
            
        self.shape                = (number_of_cells_x, number_of_cells_y, number_of_cells_z)
        self.grid                 = np.zeros(self.shape, dtype=np.uint8)
        self.history              = np.ones(self.shape, dtype=np.int64) * (-1)
        self.group_id             = np.zeros(self.shape, dtype=np.uint16)
        self.group_counter        = 0
        self.initial_seeds        = []
        self.occupied             = set()
        self.anisotropyDirections = None
        self.anisotropyStrength   = 0.0

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

    def occupy(self, x: int, y: int, z: int, epoch: int, id: int) -> None:
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
        self.group_id[x, y, z] = id
            
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

    def get_group_id(self, x: int, y: int, z: int) -> int:
        """
        Function that returns the group id of cell at coordinates (x,y,z).

        Args:
            x (int): x coord of the pixel
            y (int): y coord of the pixel
            z (int): z coord of the pixel

        Returns:
            int: group id of the selected cell.
        """
        return self.group_id[x, y, z]

    def get_group_counter(self) -> int:
        """
        Function that return the total number of crystal groups in the lattice.

        Returns:
            int: total number of crystal groups in the lattice
        """
        return self.group_counter

    def set_nucleation_seed(self, x: int, y: int, z: int, maintain_last_id: bool = False) -> None:
        """
        Function that initialize one nucleation seed.
        The function does nothing if the selected point is outside the lattice or if it's already a nucleation seed.

        Args:
            x (int): x coordinate of the nucleation seed
            y (int): y coordinate of the nucleation seed
            z (int): z coordinate of the nucleation seed
            maintain_last_id (bool, optional): if True, it gives that nucleation seed the same group id than the previous one.
                                               If self.group_counter is 0, this is bypassed to True
        """
        if self.is_point_inside(x, y, z) and (x, y, z) not in self.initial_seeds:
            if self.group_counter == 0 or not maintain_last_id:
                self.group_counter += 1
            
            self.occupy(x, y, z, epoch=0, id=self.group_counter)
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

        Returns:
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

    def setAnisotropy(self, directions: Union[np.ndarray, list], strength: float) -> None:
        """
        Initialize lattice anisotropy.

        Args:
            directions (np.array or list): iterable of vectors of length 3.
            strength (float): must be >= 0. If 0, anisotropy is disabled.
        """
        # Convert input to numpy array
        dirs = np.array(directions, dtype=float)

        if strength < 0.0:
            raise ValueError("The anisotropy strength can't be negative.")

        # Check shape: we expect an array of shape (n_dirs, 3)
        if dirs.ndim != 2 or dirs.shape[1] != 3:
            # bad shape: disable anisotropy
            self.anisotropyDirections = None
            self.anisotropyStrength = 0.0
            return

        # If strength is zero, disable anisotropy
        if strength == 0.0:
            self.anisotropyDirections = None
            self.anisotropyStrength = 0.0
            return

        # Remove zero-length directions
        norms = np.linalg.norm(dirs, axis=1)
        mask = norms > 0.0
        if not np.any(mask):
            # no valid directions -> disable anisotropy
            self.anisotropyDirections = None
            self.anisotropyStrength = 0.0
            return

        dirs = dirs[mask]
        norms = norms[mask].reshape(-1, 1)

        # Store normalized directions and strength
        self.anisotropyDirections = dirs / norms
        self.anisotropyStrength = strength

    def clearAnisotropy(self) -> None:
        """
        Disable the lattice anisotropy (removes it if was present).
        """
        self.anisotropyDirections = None
        self.anisotropyStrength = 0.0

    def computeAnisotropyWeight(self, direction: np.array) -> float:
        """
        Return the anisotropy weight for the selected direction, based on lattice anisotropy.
        
        Returns:
            (float): the weight if the anisotropy is activated, 1.0 otherwise
        """
        if len(direction) != 3:
            raise ValueError("The direction must be an array of lenght 3.")

        direction = np.array(direction, dtype=float)
        norm = np.linalg.norm(direction)
        if norm == 0.0:
            return 1.0

        if self. anisotropyDirections is not None and self.anisotropyStrength > 0.0:
            dir = direction / norm
            weights = []
            for a in self.anisotropyDirections:
                cos_t = float(np.dot(dir, a))
                if cos_t > 1.0: cos_t = 1.0
                elif cos_t < -1.0: cos_t = -1.0
                weights.append(np.exp(self.anisotropyStrength * cos_t))

            total = float(np.sum(weights))
            if total <= 0.0: return 1.0
            return total
        
        return 1.0
        



