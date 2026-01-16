import numpy as np
from classes.GrowthModel import GrowthModel
from classes.PhaseFieldLattice import PhaseFieldLattice
from classes.ParticleFlux import ParticleFlux

def laplacian(field: np.array, three_dim: bool):
    res = (np.roll(field, 1, 0) + np.roll(field, -1, 0) +
           np.roll(field, 1, 1) + np.roll(field, -1, 1) - 4 * field)
    if three_dim:
        res += np.roll(field, 1, 2) + np.roll(field, -1, 2) - 2 * field
    return res

class KobayashiGrowth(GrowthModel):
    def __init__(self, lattice: PhaseFieldLattice,
                 epsilon0: float = 0.0,
                 delta: float = 0.0,
                 n_folds: float = 0.0,
                 alpha: float = 0.0,
                 u_eq: float = 0.0,
                 tau: float = 1.0,
                 diffusivity: float = 0.0,
                 dt: float = 1e-3,
                 external_flux: ParticleFlux = None,
                 rng_seed: int = 69,
                 three_dim: bool = True,
                 verbose: bool = False)
        super().__init__(lattice, external_flux, rng_seed, three_dim, verbose)

        self.epsilon0 = epsilon0
        self.delta = delta
        self.n_folds = n_folds
        self.alpha = alpha
        self.u_eq = u_eq
        self.tau = tau
        self.D = diffusivity
        self.dt = dt
        self.prev_phi = lattice.phi

        print(self.__str__(self))

    def __str__(self):
        return f"""
        KobayashiGrowth
        -------------------------------------------------------------
        epoch={self.epoch}
        occupied={len(self.lattice.occupied)}
        epsilon0={self.epsilon0}
        delta={self.delta}
        n_folds={self.n_folds}
        alpha={self.alpha}
        u_eq={self.u_eq}
        tau={self.tau}
        D={self.D}
        dt={self.dt}
        -------------------------------------------------------------
        """

    def update_phase_field(self):
        """
        """
        phi = self.lattice.phi
        u   = self.lattice.u

        gx = (np.roll(phi, -1, 0) - np.roll(phi, 1, 0)) / 2
        gy = (np.roll(phi, -1, 1) - np.roll(phi, 1, 1)) / 2
        gz = (np.roll(phi, -1, 2) - np.roll(phi, 1, 2)) / 2 if self.three_dim else np.zeros_like(gx)

        magnitude = np.sqrt(gx**2 + gy**2 + gz**2)
        phi_az = np.arctan2(gy, gx)

        if self.three_dim:
            theta = np.arccos(np.clip(gz / magnitude, -1.0, 1.0))
            aniso = (1.0 + self.delta * np.cos(self.n_folds * phi_az) * np.sin(theta))        
        else:
            aniso = (1.0 + self.delta * np.cos(self.n_folds * phi_az))
            
        eps = self.epsilon0 * aniso
        lap_phi = laplacian(phi, self.three_dim)
        m = (self.alpha / np.pi) * np.arctan(self.alpha * (self.u_eq - u))

        dphi = (eps**2 * lap_phi + phi * (1 - phi) * (phi - 0.5 + m)) / self.tau
        self.lattice.phi += self.dt * dphi

    def update_diffusion_field(self):
        """
        """
        u   = self.lattice.u
        phi = self.lattice.phi

        du = self.D * laplacian(u, self.three_dim) + 0.5 * (phi - self.prev_phi) / self.dt
        self.lattice.u += self.dt * du
        self.prev_phi = phi.copy()

    def step(self):
        if self.verbose:
            print(f"\t\t[KobayashiGrowth] Starting evolving step {self.epoch + 1}...")

        self.update_phase_field()
        self.update_diffusion_field()

        phi = self.lattice.phi
        threshold = self.lattice.interface_threshold

        newly_solid = (phi <= threshold) & (self.lattice.history == -1)
        if np.any(newly_solid):
            coords = np.argwhere(newly_solid)
            
            for x, y, z in coords:
                self.lattice.history[x,y,z] = self.epoch
                neighbors = self.lattice.get_neighbors(x, y, z)
                assigned = False

                for nx. ny, nx in neighbors:
                    gid = self.lattice.group_id[nx, ny, nz]
                    if gid > 0:
                        self.lattice.group_id[x, y, z] = gid
                        assigned = True
                        break

                if not assigned:
                    print(f"-------------------------------------------------------------\n \
                    [KobayashiGrowth] WARNING: at step {self.epoch+1} a new nucleation has occoured at ({x}, {y}, {z})\n \
                    -------------------------------------------------------------")
                    self.lattice.group_counter += 1
                    self.lattice.group_id[x, y, z] = self.lattice.group_counter

        if self.verbose:
            print(f"\t\t[KobayashiGrowth] Finished evolving step {self.epoch + 1}!\n \
                    \t\t_____________________________________________________________")