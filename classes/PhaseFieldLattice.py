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
        self.u = np.zeros(self.shape, dtype=float)
        self.interface_threshold = interface_threshold
        self.sigma = 5.0 # TODO: decidere se tenere parametro privato o darlo come input

        print(self.__str__())

    def __str__(self):
        return super().__str__()

    def set_nucleation_seed(self, x, y, z, group_id=None):
        ndim = self.phi.ndim    
        if ndim == 3:
            if not self.is_point_inside(x, y, z):
                return
        else:
            if not self.is_point_inside(x, y):
                return

        if group_id is None:
            self.group_counter += 1
            group_id = self.group_counter

        self.initial_seeds.append((x, y) if ndim == 2 else (x, y, z))

        if ndim == 3:
            nx, ny, nz = self.phi.shape
            X, Y, Z = np.meshgrid(
                np.arange(nx),
                np.arange(ny),
                np.arange(nz),
                indexing='ij'
            )
            r2 = (X - x)**2 + (Y - y)**2 + (Z - z)**2
        else:
            nx, ny = self.phi.shape
            X, Y = np.meshgrid(
                np.arange(nx),
                np.arange(ny),
                indexing='ij'
            )
            r2 = (X - x)**2 + (Y - y)**2

        self.phi[:] += np.exp(-r2 / self.sigma**2)
        self.phi[self.phi < 1e-6] = 0.0
        solid_mask = (self.phi >= self.interface_threshold) & (self.group_id == 0)

        self.group_id[solid_mask] = group_id
        self.history[solid_mask] = 0

        print(
            "seed stats:",
            np.min(self.phi),
            np.max(self.phi),
            np.mean(self.phi),
            np.sum(self.phi > 0.5)
        )

