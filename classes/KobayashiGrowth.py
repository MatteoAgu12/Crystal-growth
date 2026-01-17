import numpy as np
from classes.GrowthModel import GrowthModel
from classes.PhaseFieldLattice import PhaseFieldLattice
from classes.ParticleFlux import ParticleFlux

class KobayashiGrowth(GrowthModel):
    def __init__(self, lattice: PhaseFieldLattice,
                 epsilon0: float = 0.0,
                 delta: float = 0.0,
                 n_folds: float = 0.0,
                 mobility: float = 0.0,
                 supersaturation: float = 0.0,
                 dt: float = 1e-3,
                 external_flux: ParticleFlux = None,
                 rng_seed: int = 69,
                 three_dim: bool = True,
                 verbose: bool = False):
        super().__init__(lattice, external_flux, rng_seed, three_dim, verbose)

        self.epsilon0 = epsilon0
        self.delta = delta
        self.n_folds = n_folds
        self.mobility = mobility
        self.supersaturation = supersaturation
        self.dt = dt
        # self.prev_phi = lattice.phi

        print(self.__str__())

    def __str__(self):
        return f"""
        KobayashiGrowth
        -------------------------------------------------------------
        epoch={self.epoch}
        epsilon0={self.epsilon0}
        delta={self.delta}
        n_folds={self.n_folds}
        mobility={self.mobility}
        supersaturation={self.supersaturation}
        dt={self.dt}
        -------------------------------------------------------------
        """

    def update_phase_field_old(self):
        """
        """
        phi = self.lattice.phi
        ndim = 3 if self.three_dim else 2

        grads = []
        for axis in range(ndim):
            grad = (np.roll(phi, -1, axis=axis) - np.roll(phi, 1, axis=axis)) / 2.0
            grads.append(grad)

        grad_sq = np.zeros_like(phi)
        for g in grads:
            grad_sq += g**2
        magnitude = np.sqrt(grad_sq + 1e-12)
        normals = [g / magnitude for g in grads]

        # anisotropy = np.zeros_like(phi)
        # for n in normals:
        #     anisotropy += n**4
        # eps = self.epsilon0 * (1.0 + self.delta * anisotropy)
        # eps2 = eps ** 2

        # TODO: per ora è solo 2D!!!!
        gx, gy = grads
        theta = np.arctan2(gy, gx)
        eps = self.epsilon0 * (1 + self.delta * np.cos(self.n_folds * theta))
        eps2 = eps**2

        div = np.zeros_like(phi)

        for axis in range(ndim):
            flux = eps2 * grads[axis]
            div += (np.roll(flux, -1, axis=axis) - np.roll(flux, 1, axis=axis)) / 2.0


        reaction_term = phi * (1.0 - phi) * (phi - 0.5 + self.supersaturation)
        phi_new = phi + self.dt * self.mobility * (div + reaction_term)
        #self.lattice.phi = np.clip(phi_new, 0.0, 1.0)
        self.lattice.phi = phi_new

        if self.epoch % 100 == 0:
            print(
                self.epoch,
                "phi max:", np.max(phi),
                "phi mean:", np.mean(phi),
                "reaction max:", np.max(phi * (1-phi) * (phi - 0.5 + self.mobility)),
                "lap max:", np.max(np.abs(div))
            )
        # phi = self.lattice.phi
        # u   = self.lattice.u
 
        # gx = (np.roll(phi, -1, 0) - np.roll(phi, 1, 0)) / 2
        # gy = (np.roll(phi, -1, 1) - np.roll(phi, 1, 1)) / 2
        # gz = (np.roll(phi, -1, 2) - np.roll(phi, 1, 2)) / 2 if self.three_dim else np.zeros_like(gx)
 
        # magnitude = np.sqrt(gx**2 + gy**2 + gz**2)
        # phi_az = np.arctan2(gy, gx)
 
        # if self.three_dim:
        #     theta = np.arccos(np.clip(gz / magnitude, -1.0, 1.0))
        #     aniso = (1.0 + self.delta * np.cos(self.n_folds * phi_az) * np.sin(theta))        
        # else:
        #     aniso = (1.0 + self.delta * np.cos(self.n_folds * phi_az))
        #     
        # eps = self.epsilon0 * aniso
        # lap_phi = laplacian(phi, self.three_dim)
        # m = (self.alpha / np.pi) * np.arctan(self.alpha * (self.u_eq - u))

        # dphi = (eps**2 * lap_phi + phi * (1 - phi) * (phi - 0.5 + m)) / self.tau
        # self.lattice.phi += self.dt * dphi

    def update_phase_field(self):
        phi = self.lattice.phi
        ndim = 3 if self.three_dim else 2

        grads = []
        for axis in range(ndim):
            grads.append((np.roll(phi, -1, axis=axis) -
                          np.roll(phi, 1, axis=axis)) / 2.0)

        # anisotropia (2D Kobayashi)
        if not self.three_dim:
            gx, gy = grads
            theta = np.arctan2(gy, gx)
            eps = self.epsilon0 * (1.0 + self.delta * np.cos(self.n_folds * theta))
        else:
            eps = self.epsilon0

        eps2 = eps**2

        div = np.zeros_like(phi)
        for axis in range(ndim):
            flux = eps2 * grads[axis]
            div += (np.roll(flux, -1, axis=axis) -
                    np.roll(flux, 1, axis=axis)) / 2.0

        double_well = phi * (1.0 - phi) * (phi - 0.5)
        driving     = self.supersaturation * phi * (1.0 - phi)
        reaction    = double_well + driving

        phi_new = phi + self.dt * self.mobility * (div + reaction)

        self.lattice.phi = np.clip(phi_new, 0.0, 1.0)

        if self.epoch % 100 == 0:
            print(self.epoch,
                  "phi max:", np.max(phi),
                  "phi mean:", np.mean(phi),
                  "reaction max:", np.max(np.abs(reaction)),
                  "lap max:", np.max(np.abs(div)))


    def step(self):
        if self.verbose:
            print(f"\t\t[KobayashiGrowth] Starting evolving step {self.epoch + 1}...")

        self.update_phase_field()

        newly_solid = (self.lattice.phi >= self.lattice.interface_threshold) & (self.lattice.history == -1)
        if np.any(newly_solid):
            coords = np.argwhere(newly_solid)
            
            for x, y, z in coords:
                self.lattice.history[x,y,z] = self.epoch
                neighbors = self.lattice.get_neighbors(x, y, z)
                assigned = False

                for nx, ny, nz in neighbors:
                    gid = self.lattice.group_id[nx, ny, nz]
                    if gid > 0:
                        self.lattice.group_id[x, y, z] = gid
                        assigned = True
                        break

                if not assigned:
                    print(f"""
                    -------------------------------------------------------------
                    [KobayashiGrowth] WARNING: at step {self.epoch+1} a new nucleation has occoured at ({x}, {y}, {z})
                    -------------------------------------------------------------""")
                    self.lattice.group_counter += 1
                    self.lattice.group_id[x, y, z] = self.lattice.group_counter

        if self.verbose:
            print(f"\t\t[KobayashiGrowth] Finished evolving step {self.epoch + 1}!\n \
                    \t\t_____________________________________________________________")




