import numpy as np
from classes.BaseLattice import BaseLattice

class PhaseFieldLattice(BaseLattice):
    def __init__(self, nx: int, ny: int, nz: int,
                 interface_threshold: float,
                 verbose: bool):
        super().__init__(nx, ny, nz, verbose)

        if interface_threshold <= 0.0 or interface_threshold > 1.0:
            raise ValueError(f"Interface threshold must be a float in range [0,1]")

        self.phi = np.zeros(self.shape, dtype=float)
        self.u = np.ones(self.shape, dtype=float)
        self.interface_threshold = interface_threshold

        print(self.__str__(self))

    def __str__(self):
        super().__str__(self)

    def set_nucleation_seed(self, x,y,z, group_id=None):
        if self.is_point_inside(x,y,z):
            if group_id is None:
                self.group_counter += 1
                group_id = self.group_counter
    
            self.initial_seeds.append((x,y,z))
            self.group_id[x,y,z] = group_id
            self.phi[x,y,z] = 1.0
            self.history[x,y,z] = 0
