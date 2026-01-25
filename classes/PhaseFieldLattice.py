import numpy as np 
from classes.BaseLattice import BaseLattice

class PhaseFieldLattice(BaseLattice):
    """
    """
    def __init__(self, nx: int, ny: int, nz: int, 
                 interface_threshold: float = 0.5, 
                 verbose: bool = False):
        if interface_threshold <= 0.0:
            raise ValueError(f"ERROR: Interface threshold must be a positive float: {interface_threshold} is not a good option.")

        super().__init__(nx, ny, nz, verbose=verbose)
    
        self.phi       = np.zeros((nx, ny, nz), dtype=np.float32)
        self.u         = np.zeros((nx, ny, nz), dtype=np.float32)
        self.grid      = np.zeros((nx, ny, nz), dtype=np.uint8)
        self.curvature = np.zeros((nx, ny, nz), dtype=np.float32)

        self.interface_threshold = interface_threshold
        self._seeds = []

        print(self.__str__())

    def __str__(self):
        return (f"[PhaseFieldLattice] size={self.shape}, "
                f"thr={self.interface_threshold}, occupied={len(self.occupied)}")

    def is_occupied(self, x: int, y: int, z: int):
        return bool(self.grid[x, y, z] == 1)

    def set_nucleation_seed(self, x: int, y: int, z: int,
                            radius: float = 4, width: float = 1.5,
                            phi_in: float = 1.0, phi_out: float = 0.0):
        """
        """
        X, Y = np.meshgrid(np.arange(self.shape[1], dtype=np.float32),
                           np.arange(self.shape[0], dtype=np.float32))
        dist = np.sqrt((Y - x) ** 2 + (X - y) ** 2)

        r = float(radius)
        w = float(max(width, 1e-6))
        prof = 0.5 * (1.0 - np.tanh((dist - r) / w))

        phi_seed = phi_out + (phi_in - phi_out) * prof
        self.phi[:, :, z] = np.maximum(self.phi[:, :, z], phi_seed.astype(np.float32))

        if (x, y, z) not in self._seeds:
            self._seeds.append((x, y, z))

        self.update_occupied_and_history(epoch=0)

        if self.verbose:
            print(f"[PhaseFieldLattice::set_nucleation_seed] seed at ({x},{y},{z}), "
                  f"r={radius}, w={smooth_width}")

    def update_occupied_and_history(self, epoch: int):
        thr = self.interface_threshold
        new_grid = (self.phi >= thr).astype(np.uint8)

        changed = (new_grid != self.grid)
        if np.any(changed):
            idx = np.argwhere(changed)
            for (x, y, z) in idx:
                if new_grid[x, y, z] == 1:
                    self.occupied.add((x, y, z))
                    if self.history[x, y, z] <= 0:
                        self.history[x, y, z] = epoch
                    
                    if len(self._seeds) > 0:
                        self.group_id[x, y, z] = self._nearest_seed_id(x, y, z)

                else:
                    if (x, y, z) in self.occupied:
                        self.occupied.remove((x, y, z))

        self.grid[:] = new_grid

    def _nearest_seed_id(self, x: int, y: int, z: int):
        best_k = 0.0
        best_d2 = None
        for k, (sx, sy, sz) in enumerate(self._seeds):
            dx = x - sx
            dy = y - sy
            dz = z - sz
            d2 = dx*dx + dy*dy + dz*dz

            if best_d2 is None or d2 < best_d2:
                best_d2 = d2
                best_k = k

        return best_k
        



