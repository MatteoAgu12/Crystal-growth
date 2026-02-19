import os
import numpy as np 
import matplotlib.pyplot as plt

from classes.BaseLattice import BaseLattice

import logging
logger = logging.getLogger("growthsim")

class PhaseFieldLattice(BaseLattice):
    """
    This class represents a specific implementation of the BaseLattice for phase-field crystal growth simulations.
    It defines the grid structure and methods for managing the phase field (phi) and concentration field (u) in addition to the occupation status of cells.
    The lattice is represented as a 3D grid, where each cell can be occupied or unoccupied. 
    It also keeps track of the history of occupation and group IDs for nucleation processes.
    """
    def __init__(self, nx: int, ny: int, nz: int, 
                 interface_threshold: float = 0.5, 
                 verbose: bool = False):
        """
        Args:
            nx (int): number of cells in the lattice along the x direction
            ny (int): number of cells in the lattice along the y direction
            nz (int): number of cells in the lattice along the z direction
            interface_threshold (float, optional): threshold value for determining occupied cells based on the phase field. Defaults to 0.5.
            verbose (bool, optional): if True, the lattice will print debug information during initialization and occupation of cells. Defaults to False.
        """
        if interface_threshold <= 0.0:
            raise ValueError(f"ERROR: Interface threshold must be a positive float: {interface_threshold} is not a good option.")

        super().__init__(nx, ny, nz, verbose=verbose)
    
        self.phi       = np.zeros((nx, ny, nz), dtype=np.float64)
        self.u         = np.zeros((nx, ny, nz), dtype=np.float64)
        self.grid      = np.zeros((nx, ny, nz), dtype=np.uint8)

        self.interface_threshold = interface_threshold
        self._seeds = []

        logger.debug("%s", self)

    def __str__(self):
        return f"""
        === PhaseFieldLattice Object ======================================== 
         * Shape:               {self.shape}
         * Nucleation seeds:    {self.initial_seeds}
         * Interface threshold: {self.interface_threshold}
         * Verbose:             {self.verbose}
        =====================================================================
        """

    def is_occupied(self, x: int, y: int, z: int) -> bool:
        """
        Check if the cell at coordinates (x,y,z) is occupied based on the phase field value and the interface threshold.

        Args:
            x (int): x coordinate of the point to check
            y (int): y coordinate of the point to check
            z (int): z coordinate of the point to check

        Returns:
            bool: returns True if the cell is occupied (phi >= interface_threshold), False otherwise
        """
        return bool(self.grid[x, y, z] == 1)

    def set_nucleation_seed(self, x: int, y: int, z: int,
                            radius: float = 5.0, width: float = 1.5,
                            phi_in: float = 1.0, phi_out: float = 0.0,
                            u_inf: float = 0.0, u_eq: float = 1.0):
        """
        Function to set a nucleation seed at coordinates (x,y,z) by initializing the phase field (phi) in a spherical region around the seed point.
        The phase field is set to phi_in inside the seed region and transitions to phi_out outside the seed region based on a hyperbolic tangent profile.

        Args:
            x (int): x coordinate of the seed point
            y (int): y coordinate of the seed point
            z (int): z coordinate of the seed point
            radius (float, optional): radius of the seed region. Defaults to 4.
            width (float, optional): width of the transition region between phi_in and phi_out. Defaults to 1.5.
            phi_in (float, optional): phase field value inside the seed region. Defaults to 1.0.
            phi_out (float, optional): phase field value outside the seed region. Defaults to 0.0.
            u_inf (float, optional): concentration field value far from the seed. Defaults to 0.0.
            u_eq (float, optional): concentration field value at equilibrium. Defaults to 1.
        """
        r = float(radius)
        w = float(max(width, 1e-6))

        if self.shape[2] <= 1:
            X, Y = np.ogrid[-x:self.shape[0]-x, -y:self.shape[1]-y]
            mask = X**2 + Y**2 <= r**2
            self.phi[:, :, z][mask] = phi_in
            self.u[:, :, z][mask] = u_eq
        else:
            X, Y, Z = np.ogrid[-x:self.shape[0]-x, -y:self.shape[1]-y, -z:self.shape[2]-z]
            mask = X**2 + Y**2 + Z**2 <= r**2
            self.phi[:, :, :][mask] = phi_in
            self.u[:, :, :][mask] = u_eq

        if (x, y, z) not in self._seeds:
            self._seeds.append((x, y, z))

        self.update_occupied_and_history(epoch=0)
        logger.debug(f"[PhaseFieldLattice::set_nucleation_seed] seed at ({x},{y},{z})")

    def update_occupied_and_history(self, epoch: int):
        """
        Update the occupation status of cells based on the current phase field values and the interface threshold.
        It also updates the history of occupation for newly occupied cells and assigns group IDs based on the nearest nucleation seed.
        
        Args:
            epoch (int): current epoch in the simulation
        """
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

    def _nearest_seed_id(self, x: int, y: int, z: int) -> int:
        """
        Find the nearest nucleation seed to the cell at coordinates (x,y,z) and return its group ID.

        Args:
            x (int): x coordinate of the cell
            y (int): y coordinate of the cell
            z (int): z coordinate of the cell

        Returns:
            int: group ID of the nearest nucleation seed (1-based index)
        """
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

        return best_k+1
        
    def save_frame(self, epoch: int, three_dim: bool, frame_dir: str, frame_list: list) -> str:
        z = 0 if self.shape[2] == 1 else self.shape[2] // 2

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        axs[0].imshow(self.phi[:,:,z].T, origin='lower', cmap='gray_r', vmin=0, vmax=1)
        axs[0].set_title(r"Crystal field ($\phi$)")

        axs[1].imshow(self.u[:,:,z].T, origin='lower', cmap='inferno', vmin=0, vmax=1)
        axs[1].set_title(r"Diffused field ($u$)")

        hist = self.history[:,:,z].astype(float)
        hist_masked = np.ma.masked_where(hist < 0, hist)
        axs[2].imshow(hist_masked.T, origin='lower', cmap='turbo')
        axs[2].set_title(r"Occupation history")

        filepath = os.path.join(frame_dir, f"frame_{epoch:05d}.png")
        plt.savefig(filepath, bbox_inches='tight')
        plt.close(fig)

        frame_list.append(filepath)


