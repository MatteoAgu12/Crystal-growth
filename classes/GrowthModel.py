from abc import ABC, abstractmethod
import numpy as np
from tqdm import tqdm

from classes.BaseLattice import BaseLattice
from classes.ParticleFlux import ParticleFlux

import logging
logger = logging.getLogger("growthsim")

class GrowthModel(ABC):
    """
    This class represents the growth model for crystal simulations.
    It defines the basic properties and methods that any specific growth model implementation should have.
    Interacts with a lattice structure (BaseLattice) and can incorporate an external particle flux (ParticleFlux).
    The growth model is responsible for performing growth steps (epochs) and updating the lattice accordingly.
    """
    def __init__(self, lattice: BaseLattice,
                 external_flux: ParticleFlux = None,
                 rng_seed: int = 69,
                 three_dim: bool = True,
                 verbose: bool = False):
        """
        Args:
            lattice (BaseLattice): the lattice structure on which the growth model will operate
            external_flux (ParticleFlux, optional): exernal particle flux to be applied during growth steps. Defaults to None.
            rng_seed (int, optional): random seed for reproducibility. Defaults to 69.
            three_dim (bool, optional): if True, the growth model will consider three-dimensional growth. Defaults to True.
            verbose (bool, optional): if True, the growth model will print debug information during growth steps. Defaults to False.
        """
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
    
    def run(self, n_steps: int, 
            callback = None, save_freq: int = 10, 
            frame_dir: str = None, frame_list: list = []):
        """
        Run the growth model for a specified number of steps (epochs).

        Args:
            n_steps (int): Number of growth steps (epochs) to perform.
            # TODO: aggiorna
        """
        with tqdm(total=n_steps, desc="Running GrowthModel", unit="epoch", disable=self.verbose) as pbar:
            for i in range(n_steps):
                self.step()
                self.epoch += 1

                if hasattr(self.lattice, "phi") and np.any(np.isnan(self.lattice.phi)):
                    if np.any(np.isnan(self.lattice.phi)):
                        logger.fatal("[GrowthModel] FATAL ERROR: NaN detected in lattice.phi at epoch %d, aborting...", self.epoch)
                        break
                    elif np.any(np.isnan(self.lattice.u)):
                        logger.fatal("[GrowthModel] FATAL ERROR: NaN detected in lattice.u at epoch %d, aborting...", self.epoch)
                        break

                if callback is not None and self.epoch % save_freq == 0:
                    callback(self.lattice, self.epoch, frame_dir, frame_list)

                pbar.update(1)
