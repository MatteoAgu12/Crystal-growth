import numpy as np
from typing import Union

class ParticleFlux:
    """
    """
    def __init__(self, flux_directions: Union[np.ndarray, list] = None, strength: float = 0.0, verbose: bool = False):
        """
        """
        self.fluxDirections = flux_directions
        self.fluxStrength   = strength
        self.verbose        = verbose
    
    def __str__(self):
        info = f"ParticleFlux object:\n \
                 Flux direction: {self.fluxDirections} \n \
                 Flux strength:  {self.fluxStrength} \n \
                 Verbose:        {self.verbose}"
        return info
    
    def set_external_flux(self, directions: Union[np.ndarray, list], strength: float):
        """
        Initialize the external diffusive flux.

        Args:
            directions (np.array or list): iterable of vectors of length 3.
            strength (float): must be >= 0. If 0, anisotropy is disabled.
        """
        dirs = np.array(directions, dtype=float)

        if strength < 0.0:
            raise ValueError("The anisotropy strength can't be negative.")

        if dirs.ndim != 2 or dirs.shape[1] != 3:
            self.fluxDirections = None
            self.fluxStrength = 0.0
            print("==========================================================================\n \
                  [ParticleFlux] WARNING: wrong inputs in ParticleFlux, no flux selected!\n \
                  ==========================================================================")
            return

        if strength == 0.0:
            self.fluxDirections = None
            self.fluxStrength = 0.0
            print("==========================================================================\n \
                  [ParticleFlux] WARNING: wrong inputs in ParticleFlux, no flux selected!\n \
                  ==========================================================================")
            return

        norms = np.linalg.norm(dirs, axis=1)
        mask = norms > 0.0
        if not np.any(mask):
            self.fluxDirections = None
            self.fluxStrength = 0.0
            print("==========================================================================\n \
                  [ParticleFlux] WARNING: no valid directions, no flux selected!\n \
                  ==========================================================================")
            return

        dirs = dirs[mask]
        norms = norms[mask].reshape(-1, 1)

        self.fluxDirections = dirs / norms
        self.fluxStrength = strength

    def clear_external_flux(self) -> None:
        """
        Disable the external diffusion flux (removes it if was present).
        """
        self.fluxDirections = None
        self.fluxStrength = 0.0

    def compute_external_flux_weights(self, direction: np.array) -> float:
        """
        Return the anisotropy weight for external flux for the selected direction, based on the flux selected.
        
        Returns:
            (float): the weight if the flux is activated, 1.0 otherwise
        """
        if len(direction) != 3:
            raise ValueError("The direction must be an array of lenght 3.")

        direction = np.array(direction, dtype=float)
        norm = np.linalg.norm(direction)
        if norm == 0.0:
            return 1.0

        if self.fluxDirections is not None and self.fluxStrength > 0.0:
            dir = direction / norm
            weights = []
            for a in self.fluxDirections:
                cos_t = float(np.dot(dir, a))
                if cos_t > 1.0: 
                    cos_t = 1.0
                elif cos_t < -1.0:
                    cos_t = -1.0
                weights.append(np.exp(self.fluxStrength * cos_t))

            total = float(np.sum(weights))
            if total <= 0.0: 
                return 1.0
            return total
        
        return 1.0
   
    
    