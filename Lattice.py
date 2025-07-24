import numpy as np

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
        self.initial_seeds = []

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
                neighbors.append([nx, ny, nz])
                
        return np.array(neighbors)

    def occupy(self, x: int, y: int, z: int) -> None:
        """
        Set the status of the cell at coordinates (x,y,z) to occupied.
        
        Args:
            x (int): x coordinate of the point to occupy
            y (int): y coordinate of the point to occupy
            z (int): z coordinate of the point to occupy
        """
        if self.is_point_inside(x, y, z):
            self.grid[x, y, z] = 1
            
    def is_occupied(self, x: int, y: int, z: int) -> bool:
        """
        Function that check if a point at coordinates (x,y,z) i occupied or not.

        Args:
            x (int): _description_
            y (int): _description_
            z (int): _description_
            
        Returns:
            (bool): 
        """
        return self.grid[x, y, z] == 1

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
            self.occupy(x, y, z)
            self.initial_seeds.append((x, y, z))
            
    def get_active_border(self) -> np.array:
        """
        Function that compute the active border of teh crystal.
        The active border is the set of all empty cells having at least one occupied cell as neighbor.

        Returnss:
            (np.array): array containing the points that form the active border
        """
        active_border = []
        occupied_sites = np.argwhere(self.grid == 1)
        for (x, y, z) in occupied_sites:
            neighbors = self.get_neighbors(x, y, z)
            
            for (x_n, y_n, z_n) in neighbors:
                if not self.is_occupied(x_n, y_n, z_n) and (x_n, y_n, z_n) not in active_border:
                    active_border.append((x_n, y_n, z_n))
        
        return np.array(active_border)
    



