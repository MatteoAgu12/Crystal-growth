import numpy as np
import itertools
from typing import Union

class Lattice:
    """
    """
    def __init__(self, number_of_cells_x: int, number_of_cells_y: int, number_of_cells_z: int, verbose: bool = False):
        """
        Args:
            number_of_cells_x (int): number of cells in the lattice along the x direction
            number_of_cells_y (int): number of cells in the lattice along the y direction
            number_of_cells_z (int): number of cells in the lattice along the z direction
        """
        if number_of_cells_x < 0 or number_of_cells_y < 0 or number_of_cells_z < 0:
            raise ValueError('ERROR: the size of the lattice must be an integer bigger or equal to zero!')
            
        self.shape                           = (number_of_cells_x, number_of_cells_y, number_of_cells_z)
        self.grid                            = np.zeros(self.shape, dtype=np.uint8)
        self.history                         = np.ones(self.shape, dtype=np.int64) * (-1)
        self.group_id                        = np.zeros(self.shape, dtype=np.uint16)
        self.group_counter                   = 0
        self.initial_seeds                   = []
        self.occupied                        = set()
        self.preferred_axes                  = []
        self.anisotropy_sticking_coefficient = 0.0
        self.anisotropy_sharpness            = 1.0
        self.anisotropy_selection_strength   = 1.0
        self.base_sticking_prob              = 0.01
        self.collect_anisotropy_stats        = False
        self.anisotropy_stats                = {"epoch" : [], "a_s" : []}
        self.verbose                         = verbose

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

    def set_miller_anisotropy(self, h: int, k: int, l: int, base_stick_prob: float = 1.0,
                              sticking_coefficient: float = 1.0, sharpness: float = 4.0, selection_strength: float = 5.0):
        """
        Define anisotropic growth based on Miller index family <hkl>.

        Args:
            h, k, l (int): Miller indices (cannot all be zero)
            sticking_coefficient (float): anisotropy strength
            sharpness (float): angular selectivity exponent
            selection_strength (float): how strong is the selection rule, goes as exp(selection_strength * anisotropy_score)
        """
        
        base = [h, k, l]
        unique_integer_dirs = set()

        for perm in set(itertools.permutations(base)):
            for signs in itertools.product([-1, 1], repeat=3):
                vec = np.array(
                    [perm[0] * signs[0],
                     perm[1] * signs[1],
                     perm[2] * signs[2]],
                    dtype=int
                )

                if np.all(vec == 0):
                    continue

                gcd = np.gcd.reduce(np.abs(vec))
                vec = vec // gcd
                unique_integer_dirs.add(tuple(vec))
    
        self.preferred_axes = [
            np.array(v, dtype=float) / np.linalg.norm(v)
            for v in unique_integer_dirs ]

        self.anisotropy_sticking_coefficient = float(sticking_coefficient)
        self.anisotropy_sharpness = float(sharpness)
        self.anisotropy_selection_strength = float(selection_strength)
        self.base_sticking_prob = 0.05
        
        # === DEBUG PRINT ===
        if self.verbose:
            print(f"\n[DEBUG LATTICE] Anisotropy Configured:")
            print(f"  - Family: <{h} {k} {l}>")
            print(f"  - Generated {len(self.preferred_axes)} axes.")
            print(f"  - Axes list: {self.preferred_axes}")
            print(f"  - Coeff: {self.anisotropy_sticking_coefficient}")
            print(f"  - Sharpness: {self.anisotropy_sharpness}")
            print(f"  - Selection strength: {self.anisotropy_selection_strength}")
            print(f"=========================================\n")
      
    def get_surface_normals(self, x : int, y : int, z : int) -> list:
        """
        Return list of unit vectors pointing from each occupied neighbor toward (x,y,z).
        No averaging here: keep the set of discrete normals.

        Args:
            x, y, z (int): empty point

        Returns:
            list: list of normal vector to the surface from each occupied neighbor
        """
        normals = []
        neighbors = self.get_neighbors(x, y, z)
        
        for n in neighbors:
            nx, ny, nz = int(n[0]), int(n[1]), int(n[2])
            if self.is_occupied(nx, ny, nz):
                v = np.array([x-nx, y-ny, z-nz], dtype=float)
                norm = np.linalg.norm(v)
                if norm > 0.0:
                    normals.append(v / norm)
                    
        return normals
    
    def anisotropy_function(self, normal: np.array) -> float:
        """
        Continuous anisotropy function a_s(n) >= 0 based on the preferred axes.

        Args:
            normals (np.array): vector normal to the surface.

        Returns:
            float: value of the anisotropy function.
        """
        if not self.preferred_axes:
            return 1.0
        
        n_norm = np.linalg.norm(normal)
        if n_norm == 0.0:
            return 0.0
        n_unit = normal / n_norm
        
        total = 0.0
        for axis in self.preferred_axes:
            total += abs(np.dot(n_unit, axis)) ** self.anisotropy_sharpness
            
        total /= max(1.0, len(self.preferred_axes))
        return 1.0 + self.anisotropy_sticking_coefficient * total
    
    def compute_structural_probability(self,x: int, y: int, z: int) -> float:
        """
        Compute a structural score for site (x,y,z) based on local discrete normals and anisotropy_function.
        Returns a non-negative score a_s >= 0.
        NOTE: this is not a probability, it is a score, it has to be treated differently depending on the simulation you are running.
        
        Args:
            x, y, z (int): coordinates of the candidate cell for the adesion.

        Returns:
            float: attachment probability
        """
        normals = self.get_surface_normals(x, y, z)
        if not normals:
            return 0.0
        
        max_a = 0.0
        for n in normals:
            a = self.anisotropy_function(n)
            if a > max_a:
                max_a = a
                
        return max_a
             
    def record_anisotropy_stats(self, sites: np.ndarray, epoch: int):
        """
        Records structural anisotropy scores for a givenlist of sites.
        Mainly for diagnositcs.

        Args:
            sites (np.ndarray): active sites
            epoch (int): epoch number
        """
        if not self.collect_anisotropy_stats:
            return
        
        values = []
        for s in sites:
            a_s = self.compute_structural_probability(s[0], s[1], s[2])
            if a_s > 0.0:
                values.append(a_s)
                
        if values:
            self.anisotropy_stats["epoch"].append(epoch)
            self.anisotropy_stats["a_s"].append(a_s)
            
    def reset_anisotropy_stats(self):
        """
        Reset the statistics about the anisotropy.
        """
        self.anisotropy_stats["epoch"] = []
        self.anisotropy_stats["a_s"] = []
        

