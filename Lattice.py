import numpy as np

class Lattice:
    def __init__(self, number_of_cells_x: int, number_of_cells_y: int, number_of_cells_z: int):
        """_summary_

        Args:
            number_of_cells_x (int): _description_
            number_of_cells_y (int): _description_
            number_of_cells_z (int): _description_
        """
        if number_of_cells_x < 0 or number_of_cells_y < 0 or number_of_cells_z < 0:
            raise ValueError('ERROR: the size of the lattice must be an integer bigger or equal to zero!')
            
        self.shape = (number_of_cells_x, number_of_cells_y, number_of_cells_z)
        self.grid = np.zeros(self.shape, dtype=np.uint8)    # 0 = empty; 1 = occupied

    def __str__(self):
        return f"Lattice has shape: {self.shape}\nNumber of occupied sites: {np.sum(self.grid)}"

    def is_point_inside(self, x: int, y: int, z: int):
        """_summary_

        Args:
            x (int): _description_
            y (int): _description_
            z (int): _description_
        """
        nx, ny, nz = self.shape
        return (0 <= x < nx) and (0 <= y < ny) and (0 <= z < nz)
        
    def get_neighbors(self, x: int, y: int, z: int):
        """_summary_

        Args:
            x (int): _description_
            y (int): _description_
            z (int): _description_
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

    def occupy(self, x: int, y: int, z: int):
        """_summary_

        Args:
            x (int): _description_
            y (int): _description_
            z (int): _description_
        """
        if self.is_point_inside(x, y, z):
            self.grid[x, y, z] = 1
            
    def is_occupied(self, x: int, y: int, z: int):
        """_summary_

        Args:
            x (int): _description_
            y (int): _description_
            z (int): _description_
        """
        return self.grid[x, y, z] == 1

