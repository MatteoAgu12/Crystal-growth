import numpy as np
from classes.PhaseFieldLattice import PhaseFieldLattice
from classes.GrowthModel import GrowthModel

import logging
logger = logging.getLogger("growthsim")

class KobayashiGrowth(GrowthModel):
    """
    This class represents a specific implementation of the GrowthModel for phase-field crystal growth simulations based on the Kobayashi model.
    It defines the growth process based on the evolution of a phase field (phi) that represents the crystal structure, and a concentration field (u) that represents the supersaturation of the system.
    The growth model interacts with a PhaseFieldLattice to manage the phase field, concentration field, and occupation status of cells.
    The Kobayashi growth model is characterized by the evolution of the phase field according to a partial differential equation that includes contributions from the gradient of the phase field, anisotropy, and a reaction term that depends on the supersaturation.
    """
    def __init__(self, lattice: PhaseFieldLattice,
                 dt: float = 0.01,
                 mobility: float = 1.0,
                 epsilon0: float = 1.0,
                 delta: float = 0.0,
                 n_folds: float = 0.0,
                 supersaturation: float = 0.0,
                 three_dim: bool = True,
                 verbose: bool = False):
        """
        Args:
            lattice (PhaseFieldLattice): the lattice structure on which the growth model will operate
            dt (float, optional): time step for the evolution of the phase field. Defaults to 0.01.
            mobility (float, optional): mobility parameter for the evolution of the phase field. Defaults to 1.0.
            epsilon0 (float, optional): base value for the anisotropy of the system. Defaults to 1.0.
            delta (float, optional): strength of the anisotropy. Defaults to 0.0.
            n_folds (float, optional): number of folds for the anisotropy. Defaults to 0.0.
            supersaturation (float, optional): supersaturation parameter for the reaction term in the phase field evolution. Defaults to 0.0.
            three_dim (bool, optional): if True, the growth model will consider three-dimensional growth. Defaults to True.
            verbose (bool, optional): if True, the growth model will print debug information during growth steps. Defaults to False.
        """
        super().__init__(lattice, three_dim=three_dim, verbose=verbose)

        self.dt = dt
        self.M = mobility
        self.epsilon0 = epsilon0
        self.delta = delta
        self.n_folds = n_folds
        self.supersaturation = supersaturation

        logger.debug("%s", self)

    def __str__(self):
        return f"""
        KobayashiGrowth
        -------------------------------------------------------------
        epoch={self.epoch}
        dt={self.dt}
        mobility={self.M}
        epsilon0={self.epsilon0}
        delta={self.delta}
        n_folds={self.n_folds}
        supersaturation={self.supersaturation}

        external_flux={self.external_flux is not None}
        three_dim={self.three_dim}
        verbose={self.verbose}
        -------------------------------------------------------------
        """

    def _step_2D(self):
        """
        Function that performs a single growth step (one epoch) of the Kobayashi growth model in 2D.
        This involves updating the phase field (phi) according to the evolution equation that includes contributions from the gradient of the phase field, anisotropy, and a reaction term that depends on the supersaturation.
        The function computes the necessary spatial derivatives and updates the phase field accordingly, while ensuring numerical stability by adjusting the time step based on the maximum value of the anisotropy.
        """
        lat = self.lattice
        z = 0 if lat.shape[-1] == 1 else (lat.shape[-1] // 2)
        
        phi = lat.phi[:, :, z]

        pad = np.pad(phi, pad_width=1, mode='edge')
        phix = 0.5 * (pad[2:, 1:-1] - pad[:-2, 1:-1])
        phiy = 0.5 * (pad[1:-1, 2:] - pad[1:-1, :-2])

        grad2 = phix * phix + phiy * phiy
        min_grad = 1e-10
        interface_mask = (phi > 1e-3) & (phi < 1.0 - 1e-3)
        mask = interface_mask & (grad2 > min_grad)
        
        theta = np.arctan2(phiy, phix)
        theta[mask] = np.arctan2(phiy[mask], phix[mask])

        eps = np.full_like(phi, self.epsilon0, dtype=np.float64)
        deps = np.zeros_like(phi, dtype=np.float64)

        if self.delta != 0.0 and self.n_folds != 0.0:
            c = np.cos(self.n_folds * theta[mask])
            s = np.sin(self.n_folds * theta[mask])

            eps[mask]  =  self.epsilon0 * (1.0 + self.delta * c)
            deps[mask] = -self.epsilon0 * self.delta * self.n_folds * s

        eps = np.maximum(eps, 1e-8)

        Jx = (eps * eps) * phix - (eps * deps) * phiy
        Jy = (eps * eps) * phiy + (eps * deps) * phix
        Jx_p = np.pad(Jx, pad_width=1, mode='edge')
        Jy_p = np.pad(Jy, pad_width=1, mode='edge')
        divJ = (0.5 * (Jx_p[2:, 1:-1] - Jx_p[:-2, 1:-1]) + 
                0.5 * (Jy_p[1:-1, 2:] - Jy_p[1:-1, :-2]))

        m = self.supersaturation
        reaction = phi * (1.0 - phi) * (phi - 0.5 + m)
        rhs = divJ + reaction
        
        eps2_max = float(np.max(eps*eps))
        dt_max = 0.20 / (self.M * (eps2_max + 1e-12))
        dt = min(self.dt, dt_max)

        phi_new = phi + (dt * self.M) * rhs
        phi_new = np.where(phi_new < -1e-3, -1e-3, phi_new)
        phi_new = np.where(phi_new > 1.0 + 1e-3, 1.0 + 1e-3, phi_new)

        lat.phi[:, :, z] = phi_new
        lat.update_occupied_and_history(epoch=self.epoch)

    def _step_3D(self):
        pass

    def step(self):
        """
        Function that performs a single growth step (one epoch) of the Kobayashi growth model. 
        This involves updating the phase field (phi) according to the evolution equation that includes contributions from the gradient of the phase field, anisotropy, and a reaction term that depends on the supersaturation.
        The function computes the necessary spatial derivatives and updates the phase field accordingly, while ensuring numerical stability by adjusting the time step based on the maximum value of the anisotropy.
        """
        logger.debug("[KobayashiGrowth] Starting epoch %d...", self.epoch + 1)

        if self.three_dim:
            self._step_3D()
        else:
            self._step_2D()

        self.lattice.update_occupied_and_history(epoch=self.epoch)

        logger.debug("[KobayashiGrowth] Finished epoch %d!", self.epoch + 1)
        logger.debug("_____________________________________________________________")


