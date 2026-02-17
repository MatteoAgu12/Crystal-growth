import numpy as np
from classes.PhaseFieldLattice import PhaseFieldLattice
from classes.GrowthModel import GrowthModel
from classes.ParticleFlux import ParticleFlux

import logging
logger = logging.getLogger("growthsim")

class StefanGrowth(GrowthModel):
    """
    """
    def __init__(self,
                 lattice: PhaseFieldLattice,
                 dt: float = 0.0001,
                 mobility: float = 1.0,
                 diffusivity: float = 0.5,
                 epsilon0: float = 0.01,
                 delta: float = 0.04,
                 n_folds: float = 6.0,
                 alpha: float = 0.9,
                 u_eq: float = 1.0,
                 latent_coeff: float = 1.6,
                 gamma: float = 10.0,
                 u_infty: float = 0.0,
                 enforce_dirichlet_u: bool = True,
                 external_flux: ParticleFlux = None,
                 three_dim: bool = False,
                 rng_seed: int = 42,
                 verbose: bool = False):
        
        super().__init__(lattice=lattice,
                         external_flux=external_flux,
                         rng_seed=rng_seed,
                         three_dim=three_dim,
                         verbose=verbose)
        
        self.lattice = lattice        
        self.dx = lattice.dx
        
        self.dt = dt
        self.diffusivity = diffusivity
        self.epsilon0 = epsilon0
        self.delta = delta
        self.n_folds = n_folds
        self.alpha = alpha
        self.u_eq = u_eq
        self.K = latent_coeff
        self.gamma = gamma
        self.u_infty = u_infty
        self.tau = 0.0003 if mobility == 0 else (1.0 / mobility) 
        
        self.phi = self.lattice.phi
        self.u   = self.lattice.u

    def _step_2D(self):
        phi = self.phi
        u = self.u
        
        p_c = phi[1:-1, 1:-1]
        p_u = phi[2:, 1:-1]
        p_d = phi[:-2, 1:-1]
        p_r = phi[1:-1, 2:]
        p_l = phi[1:-1, :-2]

        u_c = u[1:-1, 1:-1]
        u_u = u[2:, 1:-1]
        u_d = u[:-2, 1:-1]
        u_r = u[1:-1, 2:]
        u_l = u[1:-1, :-2]

        dx2 = self.dx**2
        inv_2dx = 1.0 / (2 * self.dx)

        dx_phi = (p_r - p_l) * inv_2dx
        dy_phi = (p_u - p_d) * inv_2dx

        lap_phi = (p_r + p_l + p_u + p_d - 4.0 * p_c) / dx2
        lap_u   = (u_r + u_l + u_u + u_d - 4.0 * u_c) / dx2

        theta = np.arctan2(dy_phi, dx_phi + 1e-12)        
        angle_term = np.cos(self.n_folds * theta)
        epsilon = self.epsilon0 * (1.0 + self.delta * angle_term)
        
        term_grad = (epsilon**2) * lap_phi
        m = (self.alpha / np.pi) * np.arctan(self.gamma * (self.u_eq - u_c))
        term_reaction = p_c * (1.0 - p_c) * (p_c - 0.5 + m)
        dphi_dt = (term_grad + term_reaction) / self.tau
        
        du_dt = (self.diffusivity * lap_u) + (self.K * dphi_dt)
        
        noise = 0.0
        if self.epoch % 10 == 0:
             noise = np.random.normal(0, 0.001, p_c.shape) * p_c * (1.0 - p_c)

        phi[1:-1, 1:-1] += (dphi_dt * self.dt) + noise
        u[1:-1, 1:-1]   += du_dt * self.dt

        np.clip(phi, 0.0, 1.0, out=phi)

    def _step_3D(self):
        pass

    def step(self):
        # if self.verbose:
        #     print(f"\t\t[StefanGrowth] Starting epoch {self.epoch + 1}...")
        logger.debug(f"[StefanGrowth] Starting epoch {self.epoch + 1}...")

        if self.three_dim:
            self._step_3D()
        else:
            self._step_2D()
        
        self.lattice.update_occupied_and_history(epoch=self.epoch)

        # if self.verbose:
        #     print(f"\t\t[StefanGrowth] Finished epoch {self.epoch + 1}!")
        logger.debug(f"[StefanGrowth] Finished epoch {self.epoch + 1}!")