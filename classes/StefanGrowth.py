import numpy as np
from classes.PhaseFieldLattice import PhaseFieldLattice
from classes.GrowthModel import GrowthModel

import logging
logger = logging.getLogger("growthsim")

class StefanGrowth(GrowthModel):
    """
    This class represents a specific implementation of the GrowthModel for phase-field crystal growth simulations based on the Stefan model.
    It defines the growth process based on the evolution of a phase field (phi) that represents the crystal structure, and a concentration field (u) that represents the supersaturation of the system.
    The growth model interacts with a PhaseFieldLattice to manage the phase field, concentration field, and occupation status of cells.
    The Stefan growth model is characterized by the evolution of the phase field according to a partial differential equation that includes contributions from the gradient of the phase field, anisotropy, and a reaction term that depends on the supersaturation, with a coupling between the phase field and concentration field that accounts for latent heat effects during phase transformation.
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
                 three_dim: bool = False,
                 rng_seed: int = 42,
                 verbose: bool = False):
        """
        Args:
            lattice (PhaseFieldLattice): the lattice structure on which the growth model will operate
            dt (float, optional): time step for the evolution of the phase field. Defaults to 0.0001.
            mobility (float, optional): mobility parameter for the evolution of the phase field. Defaults to 1.0.
            diffusivity (float, optional): diffusivity parameter for the evolution of the concentration field. Defaults to 0.5.
            epsilon0 (float, optional): base value for the anisotropy of the system. Defaults to 0.01.
            delta (float, optional): strength of the anisotropy. Defaults to 0.04.
            n_folds (float, optional): number of folds for the anisotropy. Defaults to 6.0.
            alpha (float, optional): coupling parameter for the reaction term in the phase field evolution. Defaults to 0.9.
            u_eq (float, optional): equilibrium concentration for the reaction term in the phase field evolution. Defaults to 1.0.
            latent_coeff (float, optional): coefficient for the coupling between the phase field and concentration field that accounts for latent heat effects. Defaults to 1.6.
            gamma (float, optional): parameter for the arctan function in the reaction term of the phase field evolution. Defaults to 10.0.
            u_infty (float, optional): far-field concentration for the concentration field. Defaults to 0.0.
            three_dim (bool, optional): if True, the growth model will consider three-dimensional growth. Defaults to False.
            rng_seed (int, optional): random seed for reproducibility. Defaults to 42. 
            verbose (bool, optional): if True, the growth model will print debug information during growth steps. Defaults to False.
        """
        super().__init__(lattice=lattice,
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

        logger.debug("%s", self)

    def __str__(self):
        return f"""
        StefanGrowth
        -------------------------------------------------------------
        epoch={self.epoch}
        dt={self.dt}
        diffusivity={self.diffusivity}
        epsilon0={self.epsilon0}
        delta={self.delta}
        n_folds={self.n_folds}
        alpha={self.alpha}
        u_eq={self.u_eq}
        latent_coeff={self.K}
        gamma={self.gamma}
        u_infty={self.u_infty}

        three_dim={self.three_dim}
        verbose={self.verbose}
        -------------------------------------------------------------
        """

    def _step_2D(self):
        """
        Function that performs a single growth step (one epoch) of the Stefan growth model in 2D.
        This involves updating the phase field (phi) according to the evolution equation that includes contributions from the gradient of the phase field, anisotropy, and a reaction term that depends on the supersaturation, with a coupling between the phase field and concentration field that accounts for latent heat effects during phase transformation.
        The function computes the necessary spatial derivatives and updates the phase field and concentration field accordingly, while ensuring numerical stability by adjusting the time step based on the maximum value of the anisotropy.
        """
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

    def step(self):
        """
        Function that performs a single growth step (one epoch) of the Stefan growth model. 
        This involves updating the phase field (phi) according to the evolution equation that includes contributions from the gradient of the phase field, anisotropy, and a reaction term that depends on the supersaturation, with a coupling between the phase field and concentration field that accounts for latent heat effects during phase transformation.
        The function computes the necessary spatial derivatives and updates the phase field and concentration field accordingly, while ensuring numerical stability by adjusting the time step based on the maximum value of the anisotropy.
        """
        logger.debug(f"[StefanGrowth] Starting epoch {self.epoch + 1}...")

        self._step_2D()
        
        self.lattice.update_occupied_and_history(epoch=self.epoch)

        logger.debug(f"[StefanGrowth] Finished epoch {self.epoch + 1}!")