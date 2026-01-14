from abc import ABC, abstractmethod
import numpy as np
from classes.Lattice import Lattice
from classes.ParticleFlux import ParticleFlux

class GrowthModel(ABC):
    def __init__(self, lattice: Lattice,
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
        if n_steps <= 0:
            return
    
        update_step = max(1, n_steps // 10)
        next_update = update_step

        for i in range(n_steps):
            self.step()
            self.epoch += 1

            if not self.verbose and (i + 1) >= next_update:
                percent = int(100 * (i + 1) / n_steps)
                print(f"=== Simulation completed at {percent}% ===")
                next_update += update_step
