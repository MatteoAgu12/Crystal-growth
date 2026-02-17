import numpy as np
from classes.PhaseFieldLattice import PhaseFieldLattice
from classes.GrowthModel import GrowthModel
from classes.ParticleFlux import ParticleFlux

class KobayashiGrowth(GrowthModel):
    """
    """
    def __init__(self, lattice: PhaseFieldLattice,
                 dt: float = 0.01,
                 mobility: float = 1.0,
                 epsilon0: float = 1.0,
                 delta: float = 0.0,
                 n_folds: float = 0.0,
                 supersaturation: float = 0.0,
                 external_flux: ParticleFlux = None,
                 three_dim: bool = True,
                 verbose: bool = False):
        super().__init__(lattice, three_dim=three_dim, verbose=verbose)

        self.dt = dt
        self.M = mobility
        self.epsilon0 = epsilon0
        self.delta = delta
        self.n_folds = n_folds
        self.supersaturation = supersaturation

        print(self.__str__())

    def __str__(self):
        return (f"[KobayashiGrowth] dt={self.dt}, M={self.M}, eps0={self.epsilon0}, "
                f"delta={self.delta}, n={self.n_folds}, m={self.supersaturation}, "
                f"occupied={len(self.lattice.occupied)}")

    def _step_2D(self):
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
        if self.verbose:
            print(f"\t\t[KobayashiGrowth] Starting epoch {self.epoch + 1}...")

        if self.three_dim:
            self._step_3D()
        else:
            self._step_2D()

        self.lattice.update_occupied_and_history(epoch=self.epoch)

        if self.verbose:
            print(f"\t\t[KobayashiGrowth] Finished epoch {self.epoch + 1}!\n \
                    _____________________________________________________________")


