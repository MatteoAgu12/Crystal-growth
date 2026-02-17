from abc import ABC, abstractmethod
import numpy as np
from tqdm import tqdm

from classes.BaseLattice import BaseLattice
from classes.ParticleFlux import ParticleFlux

import logging
logger = logging.getLogger("growthsim")

class GrowthModel(ABC):
    def __init__(self, lattice: BaseLattice,
                 external_flux: ParticleFlux = None,
                 rng_seed: int = 69,
                 three_dim: bool = True,
                 verbose: bool = False):
        self.lattice       = lattice
        self.epoch         = 0
        self.external_flux = external_flux
        self.rng           = np.random.default_rng(rng_seed)
        self.three_dim     = three_dim
        self.verbose       = verbose
    
    def __str__(self):
        return f"""
        {self.__class__.__name__}
        -------------------------------------------------------------
        epoch={self.epoch}
        occupied={len(self.lattice.occupied)}
        -------------------------------------------------------------
        """
    
    @abstractmethod
    def step(self):
        """
        Perform a single growth step (one epoch).
        Must be implemented by subclasses.
        """
        pass
    
    def run(self, n_steps: int):
        with tqdm(total=n_steps, desc="Running GrowthModel", unit="epoch", disable=self.verbose) as pbar:
            for i in range(n_steps):
                self.step()
                self.epoch += 1

                if hasattr(self.lattice, "phi") and np.any(np.isnan(self.lattice.phi)):
                    if np.any(np.isnan(self.lattice.phi)):
                        # print(f"[GrowthModel] FATAL ERROR: NaN detected in lattice.phi at epoch {self.epoch}, aborting...")
                        logger.fatal("[GrowthModel] FATAL ERROR: NaN detected in lattice.phi at epoch %d, aborting...", self.epoch)
                        break
                    elif np.any(np.isnan(self.lattice.u)):
                        # print(f"[GrowthModel] FATAL ERROR: NaN detected in lattice.u at epoch {self.epoch}, aborting...")
                        logger.fatal("[GrowthModel] FATAL ERROR: NaN detected in lattice.u at epoch %d, aborting...", self.epoch)
                        break

                pbar.update(1)
