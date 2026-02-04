import numpy as np
from classes.PhaseFieldLattice import PhaseFieldLattice
from classes.GrowthModel import GrowthModel
from classes.ParticleFlux import ParticleFlux

class StefanGrowth(GrowthModel):
    """
    """
    def __init__(self,
                 lattice: PhaseFieldLattice,
                 dt: float = 1e-4,
                 mobility: float = 1.0,
                 diffusivity: float = 1.0,
                 epsilon0: float = 1.0,
                 delta: float = 0.0,
                 n_folds: float = 0.0,
                 alpha: float = 1.0,
                 u_eq: float = 0.0,
                 latent_coeff: float = 0.5,
                 u_infty: float = 0.0,
                 enforce_dirichlet_u: bool = False,
                 external_flux: ParticleFlux = None,
                 three_dim: bool = False,
                 rng_seed: int = 69,
                 verbose: bool = False):
        super().__init__(lattice=lattice,
                         external_flux=external_flux,
                         rng_seed=rng_seed,
                         three_dim=three_dim,
                         verbose=verbose)

        self.dt = dt
        self.M = mobility
        self.D = diffusivity

        self.epsilon0 = epsilon0
        self.delta = delta
        self.n_folds = n_folds

        self.L = latent_coeff
        self.alpha = alpha
        self.u_eq = u_eq
        self.u_infty = u_infty
        self.enforce_dirichlet_u = enforce_dirichlet_u

    def __str__(self) -> str:
        return (f"[StefanGrowth] dt={self.dt}, M={self.M}, D={self.D}, "
                f"eps0={self.epsilon0}, delta={self.delta}, n_folds={self.n_folds}, "
                f"alpha={self.alpha}, u_eq={self.u_eq}, L={self.L}, "
                f"u_infty={self.u_infty}, dirichlet_u={self.enforce_dirichlet_u}")

    @staticmethod
    def _laplacian_2d_old(field2d: np.ndarray) -> np.ndarray:
        pad = np.pad(field2d, pad_width=1, mode='edge')
        return (pad[2:, 1:-1] + pad[:-2, 1:-1] + pad[1:-1, 2:] + pad[1:-1, :-2]
                - 4.0 * pad[1:-1, 1:-1])

    def _laplacian_2d(self, field2d: np.ndarray) -> np.ndarray:
        pad = np.pad(field2d, 1, mode='constant', constant_values=self.u_infty)
        return (pad[2:,1:-1] + pad[:-2,1:-1] + pad[1:-1,2:] + pad[1:-1,:-2]
                - 4.0*pad[1:-1,1:-1])


    def _apply_dirichlet_u(self, u2d: np.ndarray) -> None:
        u2d[0, :] = self.u_infty
        u2d[-1, :] = self.u_infty
        u2d[:, 0] = self.u_infty
        u2d[:, -1] = self.u_infty

    def _step_2D(self):
        lat = self.lattice
        z = int(lat.shape[2] / 2)
        phi = lat.phi[:, :, z].astype(np.float64, copy=False)
        u = lat.u[:, :, z].astype(np.float64, copy=False)

        # === phi: anisotropic flux + local driving ===
        pad_phi = np.pad(phi, pad_width=1, mode='edge')
        phix = 0.5 * (pad_phi[2:, 1:-1] - pad_phi[:-2, 1:-1])
        phiy = 0.5 * (pad_phi[1:-1, 2:] - pad_phi[1:-1, :-2])

        grad2 = phix * phix + phiy * phiy
        min_grad = 1e-10
        interface_mask = (phi > 1e-3) & (phi < 1.0 - 1e-3)
        mask = interface_mask & (grad2 > min_grad)

        theta = np.arctan2(phiy, phix)

        eps = np.full_like(phi, self.epsilon0, dtype=np.float64)
        deps = np.zeros_like(phi, dtype=np.float64)

        if self.delta != 0.0 and self.n_folds != 0.0:
            c = np.cos(self.n_folds * theta[mask])
            s = np.sin(self.n_folds * theta[mask])
            eps[mask] = self.epsilon0 * (1.0 + self.delta * c)
            deps[mask] = -self.epsilon0 * self.delta * self.n_folds * s

        eps = np.maximum(eps, 1e-8)

        # Anisotropic flux (same structure as KobayashiGrowth)
        Jx = (eps * eps) * phix - (eps * deps) * phiy
        Jy = (eps * eps) * phiy + (eps * deps) * phix

        Jx_p = np.pad(Jx, pad_width=1, mode='edge')
        Jy_p = np.pad(Jy, pad_width=1, mode='edge')
        divJ = (0.5 * (Jx_p[2:, 1:-1] - Jx_p[:-2, 1:-1]) +
                0.5 * (Jy_p[1:-1, 2:] - Jy_p[1:-1, :-2]))

        # Driving from the diffusive field u
        m = self.alpha * (self.u_eq - u)

        # test: noise
        noise_amp = 0.1
        eta = self.rng.normal(0.0, 1.0, size=phi.shape)
        eta *= (phi * (1.0 - phi))  # only at the interface
        # test: noise

        reaction = phi * (1.0 - phi) * (phi - 0.5 + m) + noise_amp * eta
        rhs_phi = divJ + reaction

        dt = self.dt
        phi_new = phi + (dt * self.M) * rhs_phi

        # mild clamping for numerical stability
        phi_new = np.where(phi_new < -1e-3, -1e-3, phi_new)
        phi_new = np.where(phi_new > 1.0 + 1e-3, 1.0 + 1e-3, phi_new)

        # curvature proxy: Laplacian(phi)
        curvature = (pad_phi[2:, 1:-1] + pad_phi[:-2, 1:-1] +
                     pad_phi[1:-1, 2:] + pad_phi[1:-1, :-2] -
                     4.0 * pad_phi[1:-1, 1:-1])

        # === u: diffusion + Stefan-like source from phase change ===
        lap_u = self._laplacian_2d(u)
        u_new = u + dt * self.D * lap_u + self.L * (phi_new - phi)

        if self.enforce_dirichlet_u:
            self._apply_dirichlet_u(u_new)

        lat.phi[:, :, z] = phi_new.astype(np.float32)
        lat.u[:, :, z] = u_new.astype(np.float32)
        lat.curvature[:, :, z] = curvature.astype(np.float32)

        # tmp
        # if self.epoch % 100 == 0:
        #     interface = (phi > 1e-3) & (phi < 1-1e-3)
        #     print("m_interface_mean =", np.mean(m[interface]))


    def step(self):
        if self.verbose:
            print(f"\t\t[StefanGrowth] Starting epoch {self.epoch + 1}...")

        if self.three_dim:
            raise NotImplementedError("StefanGrowth currently supports only 2D simulations.")

        else:
            self._step_2D()
        
        self.lattice.update_occupied_and_history(epoch=self.epoch)

        if self.verbose:
            print(f"\t\t[StefanGrowth] Finished epoch {self.epoch + 1}!")
