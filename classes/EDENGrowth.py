import numpy as np
from typing import Union
from classes.KineticLattice import KineticLattice
from classes.ParticleFlux import ParticleFlux
from classes.GrowthModel import GrowthModel

import logging
logger = logging.getLogger("growthsim")

class EDENGrowth(GrowthModel):
    """
    This class represents a specific implementation of the GrowthModel for Eden crystal growth simulations.
    It defines the growth process based on the random selection of sites from the active border of the existing crystal structure, where new particles are added to the crystal.
    The growth model interacts with a KineticLattice to manage the occupation of cells and can incorporate an external particle flux to influence the growth process.
    The Eden growth model is characterized by the addition of new particles to the crystal structure at sites that are adjacent to already occupied sites, with the selection of these sites being random and potentially biased by an external flux.
    """
    def __init__(self, lattice: KineticLattice,
                 external_flux: ParticleFlux = None, 
                 rng_seed:int = 69, 
                 three_dim: bool = True, 
                 verbose: bool = False):
        """
        Args:
            lattice (KineticLattice): the lattice structure on which the growth model will operate
            external_flux (ParticleFlux, optional): exernal particle flux to be applied during growth steps. Defaults to None.
            rng_seed (int, optional): random seed for reproducibility. Defaults to 69.
            three_dim (bool, optional): if True, the growth model will consider three-dimensional growth. Defaults to True.
            verbose (bool, optional): if True, the growth model will print debug information during growth
        """                 
        super().__init__(lattice, rng_seed, three_dim, verbose)
        self.external_flux = external_flux
        
    def __str__(self):
        return super().__str__()
    
    def _choose_random_border_site(self, active_border: np.ndarray, reference_point: np.ndarray = None) -> Union[np.ndarray, None]:
        """
        This function randomly selects a site from the active border.
        If a lattice with anisotropy and/or an external flux is provided, selection is biased accordingly.

        Args:
            active_border (np.ndarray): active border of the crystal, obtained via Lattoce.get_active_border().
            reference_point (np.ndarray, optional): point with respect to compute the anisotropy weights, default to None.

        Returns:
            (np.ndarray): coordinates of the randomly selected site of the active border.
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
        logger.debug(f"[EDENGrowth::_choose_random_border_site]: new particle in ({site[0], site[1], site[2]})")
        
        return site
    
    def step(self):
        """
        Function that performs a single growth step (one epoch) of the Eden growth model. 
        This involves randomly selecting a site from the active border of the existing crystal structure and adding a new particle to the crystal at that site, with the selection potentially biased by an external flux.
        If no active border is found, the function will simply return without modifying the lattice.
        """
        logger.debug(f"[EDENGrowth] Starting epoch {self.epoch + 1}...")

        candidates = self.lattice.get_active_border()
        if candidates.size == 0:
            logger.warning("[EDENGrowth] WARNING: at step %d no active border has been found!", self.epoch)
            return
        
        (xmin, xmax), (ymin, ymax), (zmin, zmax) = self.lattice.get_crystal_bounding_box()
        reference_point = np.array([0.5 * (xmin+xmax), 0.5 * (ymin+ymax), 0.5 * (zmin+zmax)])
        new_site = self._choose_random_border_site(candidates, reference_point=reference_point)
        
        for nx, ny, nz in self.lattice.get_neighbors(*new_site):
            if self.lattice.is_occupied(nx, ny, nz):
                gid = self.lattice.get_group_id(nx, ny, nz)
                self.lattice.occupy(*new_site, epoch=self.epoch, id=gid)
    
                logger.debug("[EDENGrowth] Attached at %s (Id: %d)", new_site, gid)
                logger.debug("[EDENGrowth] Finished epoch %d!", self.epoch + 1)
                logger.debug("_____________________________________________________________")
                return



