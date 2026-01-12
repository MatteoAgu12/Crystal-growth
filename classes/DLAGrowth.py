import numpy as np
from Lattice import Lattice
from ParticleFlux import ParticleFlux
from GrowthModel import GrowthModel

class DLAGrowth(GrowthModel):
    def __init__(self, lattice: Lattice,
                 n_particles: int,
                 generation_padding: int,
                 outer_limit_padding: int,
                 external_flux: ParticleFlux = None, 
                 rng_seed: int = 69, 
                 three_dim: bool = True,
                 verbose: bool = False):
        super().__init__(lattice, external_flux, rng_seed, three_dim, verbose)
        
        if n_particles <= 0:
            raise ValueError("[DLAGrowth::generate_random_point_on_box] ERROR: number of particles must be > 0. Aborted.")

        if outer_limit_padding <= generation_padding:
            raise ValueError("[DLAGrowth::generate_random_point_on_box] ERROR: outer limit padding must be > generation padding. Aborted.")

        self.n_particles = n_particles
        self.generation_padding = generation_padding
        self.outer_limit_padding = outer_limit_padding
        
        self.steps = []
        self.restarts = []
    
    def __str__(self):
        return super().__str__()
    
    @staticmethod
    def _generate_random_point_on_box(self, bounding_box: tuple) -> np.array:
        """
        This function randomly generates a point on the surface of a box.
        It randomly selects one face of the box, and then it generates a random point constrained on that face

        Args:
            bounding_box (tuple): bounding box on which generate the point.

        Raises:
            ValueError: if the input parameter 'bounding_box' is not an object of lenght == 3, the program stops.

        Returns:
            (np.array): the coordinates of the randomly generated point
        """
        if len(bounding_box) != 3:
            raise ValueError(f"[DLAGrowth::generate_random_point_on_box] FATAL ERROR: \
                in function 'generate_random_point_on_sphere' the center input parameter \
                must be the set of coordinates of the center of the sphere. \
                \nThe value {bounding_box} has been inserted.\n Aborted.")

        axis = self.rng.integers(0, 3)
        face = self.rng.integers(0, 2)
        point = np.zeros(3, dtype=int)

        for d in range(3):
            if d == axis:
                point[d] = int(bounding_box[d][face])
            else:
                point[d] = int(self.rng.integers(bounding_box[d][0], bounding_box[d][1] + 1))

        return point
    
    @staticmethod
    def _particle_random_walk(self, initial_coordinate: np.array, outer_allowed_bounding_box: tuple,
                              max_steps: int = 100):
        """
        This function creates a particle in position 'initial_coordinate' and performes a random walk, weighted by the anisotropy factor and external flux.
        If the particles arrives in a site with an occupied neighbor, it stops and becomes part of the crystal.
        If the particle exits from the bounding box 'outer_allowed_bounding_box', its position is set to the initial one and the random walk restarts.

        Args:
            lattice (Lattice): custom Lattice object.
            initial_coordinate (np.array): coordinates of the spawn point of the new particle.
            outer_bounding_box (tuple): bounding box outside which the particle can't go. If it happens, the random walk restarts.
            epoch (int): current epoch number.
            max_steps (int, optional): maximum number of step performed before restarting the walk.
            three_dim (bool, optional): decides if the crystal is two or three dimentional. Defaults to True.
        """
        position = initial_coordinate.copy()
        total_steps = 0
        restarts = 0

        if self.three_dim:
            candidate_steps = np.array(
                [[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]],
                dtype=int
            )
        else:
            candidate_steps = np.array(
                [[1,0,0],[-1,0,0],[0,1,0],[0,-1,0]],
                dtype=int
            )

        while True:
            total_steps += 1
            if self.external_flux is not None:
                weights = np.array([
                    self.external_flux.weight(step) for step in candidate_steps
                ])
                if weights.sum() > 0:
                    idx = self.rng.choice(len(candidate_steps), p=weights/weights.sum())
                else:
                    idx = self.rng.integers(len(candidate_steps))
            else:
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

            # ---- sticking ----
            for nx, ny, nz in self.lattice.get_neighbors(*position):
                if self.lattice.is_occupied(nx, ny, nz):
                    n = self.lattice.get_surface_normal(*position)
                    a = self.lattice.compute_structural_probability(n)

                    p = min(1.0, a)
                    if self.rng.random() > p:
                        break

                    gid = self.lattice.get_group_id(nx, ny, nz)
                    self.lattice.occupy(*position, epoch=self.epoch, id=gid)
                    
                    if self.verbose:
                        print(f"[DLAGrowth::_particle_random_walk] for epoch {self.epoch+1}: \
                                total steps = {total_steps} | restarts = {restarts}")
                        print(f"\t\tParticle attached in ({position[0], position[1], position[2]})")
                    return
        
    def step(self):
        generation_box = self.lattice.get_crystal_bounding_box(
            padding=self.generation_padding
        )
        outer_box = self.lattice.get_crystal_bounding_box(
            padding=self.outer_limit_padding
        )

        start = self._random_point_on_box(generation_box, self.rng)
        steps, restarts = self._particle_random_walk(start, outer_box)

        self.steps.append(steps)
        self.restarts.append(restarts)




