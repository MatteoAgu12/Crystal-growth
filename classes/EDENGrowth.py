import numpy as np
from typing import Union
from classes.KineticLattice import KineticLattice
from classes.ParticleFlux import ParticleFlux
from classes.GrowthModel import GrowthModel

class EDENGrowth(GrowthModel):
    def __init__(self, lattice: KineticLattice,
                 external_flux: ParticleFlux = None, 
                 rng_seed:int = 69, 
                 three_dim: bool = True, 
                 verbose: bool = False):
        super().__init__(lattice, external_flux, rng_seed, three_dim, verbose)
        
    def __str__(self):
        return super().__str__()
    
    def _choose_random_border_site(self, active_border: np.array, reference_point: np.array = None) -> Union[np.array, None]:
        """
        This function randomly selects a site from the active border.
        If a lattice with anisotropy and/or an external flux is provided, selection is biased accordingly.

        Args:
            active_border (np.array): active border of the crystal, obtained via Lattoce.get_active_border().
            reference_point (np.array, optional): point with respect to compute the anisotropy weights, default to None.

        Returns:
            (np.array): coordinates of the randomly selected site of the active border.
        """
        if reference_point is not None:
            if len(reference_point) != 3 and reference_point.ndim != 1:
                raise ValueError("[EDENGrowth::_choose_random_border_site] FATAL ERROR: \
                                  the reference point must be in the form (x,y,z)")
    
        if len(active_border) == 0: 
            return None

        weights = np.ones(len(active_border), dtype=float)
        if self.external_flux is not None:
            for i, site in enumerate(active_border):
                direction_vec = site - reference_point
                weights[i] = self.external_flux.compute_external_flux_weights(direction_vec)

        weights = np.array(weights)
        weights /= np.sum(weights)
        idx = self.rng.choice(len(active_border), p=weights)
        site = active_border[idx].astype(int)
        
        if self.verbose:
            print(f"\t\t[EDENGrowth::_choose_random_border_site]: new particle in ({site[0], site[1], site[2]})")
        
        return site
    
    def step(self):
        if self.verbose:
            print(f"\t\t[EDENKineticGrowth] Starting epoch {self.epoch + 1}...")

        candidates = self.lattice.get_active_border()
        if candidates.size == 0:
            print(f"""
            ##############################################################
            [EDENKineticGrowth] WARNING: at step {self.epoch} no active border has been found!
            Skipped...
            ##############################################################""")
            return
        
        (xmin, xmax), (ymin, ymax), (zmin, zmax) = self.lattice.get_crystal_bounding_box()
        reference_point = np.array([0.5 * (xmin+xmax), 0.5 * (ymin+ymax), 0.5 * (zmin+zmax)])
        new_site = self._choose_random_border_site(candidates, reference_point=reference_point)
        
        for nx, ny, nz in self.lattice.get_neighbors(*new_site):
            if self.lattice.is_occupied(nx, ny, nz):
                gid = self.lattice.get_group_id(nx, ny, nz)
                self.lattice.occupy(*new_site, epoch=self.epoch, id=gid)
    
                if self.verbose:
                    print(f"\t\t[EDENKineticGrowth] Attached at {new_site} (Id: {gid})")
                    print(f"\t\t[EDENKineticGrowth] Finished epoch {self.epoch + 1}!\n \
                            _____________________________________________________________")
                return



