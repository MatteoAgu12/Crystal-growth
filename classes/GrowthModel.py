from abc import ABC, abstractmethod
import numpy as np
from Lattice import Lattice
from ParticleFlux import ParticleFlux

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
        return (
            f"{self.__class__.__name__} | "
            f"epoch={self.epoch} | "
            f"occupied={len(self.lattice.occupied)}"
    )
    
    @abstractmethod
    def step(self):
        """
        Perform a single growth step (one epoch).
        Must be implemented by subclasses.
        """
        pass
    
    def run(self, n_steps: int):
        if self.verbose:
                print(f"=== [GrowthModel] STARTING THE SIMULATION ========")
                
        for i in range(n_steps):
            if self.verbose:
                print(f"[GrowthModel] Starting step {i+1}...")
                
            self.step()
            self.epoch += 1
            
            if self.verbose:
                print(f"[GrowthModel] Performed step {i+1}.")
            else:
                if (i / n_steps) % 10 == 0:
                    print(f"[GrowthModel] Simulation at {int((i / n_steps) * 10)}%.")
                    
        if self.verbose:
                print(f"=== [GrowthModel] SIMULATION COMPLETED ==========")