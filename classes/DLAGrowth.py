import numpy as np
from classes.KineticLattice import KineticLattice
from classes.ParticleFlux import ParticleFlux
from classes.GrowthModel import GrowthModel

class DLAGrowth(GrowthModel):
    def __init__(self, lattice: KineticLattice,
                 generation_padding: int,
                 outer_limit_padding: int,
                 external_flux: ParticleFlux = None, 
                 rng_seed: int = 69, 
                 three_dim: bool = True,
                 verbose: bool = False):
        super().__init__(lattice, external_flux, rng_seed, three_dim, verbose)
        
        if outer_limit_padding <= generation_padding:
            raise ValueError("[DLAGrowth] ERROR: outer limit padding must be > generation padding. Aborted.")
            
        self.generation_padding = generation_padding
        self.outer_limit_padding = outer_limit_padding
        
        self.steps = []
        self.restarts = []

        print(self.__str__())
    
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

    def _generate_random_point_on_box(self, bounding_box: list) -> np.array:
        """
        Generates a random point on the surface of the bounding box.
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
    
    def _particle_random_walk(self, initial_coordinate: np.array, outer_allowed_bounding_box: list,
                              max_steps: int = 5000):
        """
        Performs the random walk for a single particle.
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

                    if self.verbose:
                        print(f"\t\t\t[DLAGrowth] Attached at {position} (Steps: {total_steps}, Restarts: {restarts})")
                    return

    def step(self):
        if self.verbose:
            print(f"\t\t[DLAGrowth] Starting epoch {self.epoch + 1}...")

        generation_box = self.lattice.get_crystal_bounding_box(padding=self.generation_padding)
        outer_box = self.lattice.get_crystal_bounding_box(padding=self.outer_limit_padding)

        # Genera e muovi la particella
        start = self._generate_random_point_on_box(generation_box)
        self._particle_random_walk(start, outer_box)

        if self.verbose:
            print(f"\t\t[DLAGrowth] Finished epoch {self.epoch + 1}!\n \
                    \t\t_____________________________________________________________")