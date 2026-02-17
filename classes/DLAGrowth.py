import numpy as np
from classes.KineticLattice import KineticLattice
from classes.ParticleFlux import ParticleFlux
from classes.GrowthModel import GrowthModel

import logging
logger = logging.getLogger("growthsim")

class DLAGrowth(GrowthModel):
    """
    This class represents a specific implementation of the GrowthModel for Diffusion Limited Aggregation (DLA) crystal growth simulations.
    It defines the growth process based on random walks of particles that stick to the existing crystal structure when they come into contact with it.
    The growth model interacts with a KineticLattice to manage the occupation of cells and can incorporate an external particle flux to influence the growth process.
    The DLA growth model is characterized by the generation of particles at a certain distance from the existing crystal and allowing them to perform a random walk until they either stick to the crystal or move too far away, at which point they are regenerated.
    """
    def __init__(self, lattice: KineticLattice,
                 generation_padding: int,
                 outer_limit_padding: int,
                 external_flux: ParticleFlux = None, 
                 rng_seed: int = 69, 
                 three_dim: bool = True,
                 verbose: bool = False):
        """
        Args:
            lattice (KineticLattice): the lattice structure on which the growth model will operate
            generation_padding (int): distance from the existing crystal at which new particles are generated
            outer_limit_padding (int): distance from the existing crystal beyond which particles are regenerated if they move too far away
            external_flux (ParticleFlux, optional): exernal particle flux to be applied during growth steps. Defaults to None.
            rng_seed (int, optional): random seed for reproducibility. Defaults to 69.
            three_dim (bool, optional): if True, the growth model will consider three-dimensional growth. Defaults to True.
            verbose (bool, optional): if True, the growth model will print debug information during growth steps. Defaults to False.
        """
        super().__init__(lattice, external_flux, rng_seed, three_dim, verbose)
        
        if outer_limit_padding <= generation_padding:
            raise ValueError("[DLAGrowth] ERROR: outer limit padding must be > generation padding. Aborted.")
            
        self.generation_padding = generation_padding
        self.outer_limit_padding = outer_limit_padding
        
        self.steps = []
        self.restarts = []

        logger.debug("%s", self)
    
    def __str__(self):
        return f"""
        DLAGrowth
        -------------------------------------------------------------
        epoch={self.epoch}
        occupied={len(self.lattice.occupied)}
        generation padding={self.generation_padding}
        outer_padding={self.outer_limit_padding}
        -------------------------------------------------------------
        """

    def _generate_random_point_on_box(self, bounding_box: list) -> np.ndarray:
        """
        Generates a random point on the surface of the bounding box.

        Args:
            bounding_box (list): list of tuples representing the min and max coordinates of the bounding box in the format [(xmin, xmax), (ymin, ymax), (zmin, zmax)]

        Returns:
            np.ndarray: a point on the surface of the bounding box
        """
        if len(bounding_box) != 3:
            raise ValueError(f"[DLAGrowth] FATAL ERROR: \
                               bounding box must have length 3.")

        axis = self.rng.integers(0, 3)
        face = self.rng.integers(0, 2)
        point = np.zeros(3, dtype=int)

        for d in range(3):
            if d == axis:
                point[d] = int(bounding_box[d][face])
            else:
                point[d] = int(self.rng.integers(bounding_box[d][0], bounding_box[d][1] + 1))

        return point
    
    def _particle_random_walk(self, initial_coordinate: np.ndarray, outer_allowed_bounding_box: list,
                              max_steps: int = 5000):
        """
        Performs the random walk for a single particle.

        Args:
            initial_coordinate (np.ndarray): the starting coordinate of the particle
            outer_allowed_bounding_box (list): list of tuples representing the min and max coordinates of the bounding box in the format [(xmin, xmax), (ymin, ymax), (zmin, zmax)] beyond which the particle will be regenerated if it moves too far away.
            max_steps (int, optional): maximum number of steps a particle can take before being regenerated. Defaults to 5000.

        Raises:
            ValueError: if the initial coordinate is not within the outer allowed bounding box, the function raises an error.
        """
        position = initial_coordinate.copy()
        total_steps = 0
        restarts = 0

        if self.three_dim:
            candidate_steps = np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]], dtype=int)
        else:
            candidate_steps = np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0]], dtype=int)

        while True:
            total_steps += 1
            idx = self.rng.integers(len(candidate_steps))

            position += candidate_steps[idx]

            (xmin, xmax), (ymin, ymax), (zmin, zmax) = outer_allowed_bounding_box
            if not (xmin <= position[0] <= xmax and
                    ymin <= position[1] <= ymax and
                    zmin <= position[2] <= zmax) or total_steps > max_steps:
                position = initial_coordinate.copy()
                restarts += 1
                total_steps = 0
                continue

            neighbors = self.lattice.get_neighbors(*position)
            for nx, ny, nz in neighbors:
                if self.lattice.is_occupied(nx, ny, nz):

                    gid = self.lattice.get_group_id(nx, ny, nz)
                    self.lattice.occupy(*position, epoch=self.epoch, id=gid)
                    logger.debug(f"[DLAGrowth] Attached at {position} (Steps: {total_steps}, Restarts: {restarts})")
                    return

    def step(self):
        """
        Perform a single growth step (one epoch) of the DLA growth model. 
        This involves generating a new particle at a random point on the surface of a bounding box around the existing crystal, 
        and allowing it to perform a random walk until it either sticks to the crystal or moves too far away, 
        at which point it is regenerated.
        """
        logger.debug(f"[DLAGrowth] Starting epoch {self.epoch + 1}...")

        generation_box = self.lattice.get_crystal_bounding_box(padding=self.generation_padding)
        outer_box = self.lattice.get_crystal_bounding_box(padding=self.outer_limit_padding)

        start = self._generate_random_point_on_box(generation_box)
        self._particle_random_walk(start, outer_box)

        logger.debug("\t\t[DLAGrowth] Finished epoch %d! \
                      \t\t_____________________________________________________________", self.epoch + 1)