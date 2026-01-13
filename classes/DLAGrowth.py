import numpy as np
from classes.Lattice import Lattice
from classes.ParticleFlux import ParticleFlux
from classes.GrowthModel import GrowthModel

class DLAGrowth(GrowthModel):
    def __init__(self, lattice: Lattice,
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
        
        # Statistiche diagnostiche
        self.steps = []
        self.restarts = []
    
    def _generate_random_point_on_box(self, bounding_box: list) -> np.array:
        """
        Generates a random point on the surface of the bounding box.
        Removed @staticmethod because it uses self.rng.
        """
        if len(bounding_box) != 3:
            raise ValueError(f"[DLAGrowth] FATAL ERROR: bounding box must have length 3.")

        # Scegliamo un asse (0=x, 1=y, 2=z) e una faccia (0=min, 1=max)
        axis = self.rng.integers(0, 3)
        face = self.rng.integers(0, 2)
        point = np.zeros(3, dtype=int)

        for d in range(3):
            if d == axis:
                # Fissiamo la coordinata sull'asse scelto (faccia del cubo)
                point[d] = int(bounding_box[d][face])
            else:
                # Le altre coordinate sono random entro i limiti
                point[d] = int(self.rng.integers(bounding_box[d][0], bounding_box[d][1] + 1))

        return point
    
    def _particle_random_walk(self, initial_coordinate: np.array, outer_allowed_bounding_box: list,
                              max_steps: int = 5000):
        """
        Performs the random walk for a single particle.
        Removed @staticmethod because it uses self.lattice and self.rng.
        """
        position = initial_coordinate.copy()
        total_steps = 0
        restarts = 0

        # Definizione passi possibili
        if self.three_dim:
            candidate_steps = np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]], dtype=int)
        else:
            candidate_steps = np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0]], dtype=int)

        while True:
            total_steps += 1
            
            # --- 1. Scelta del passo (Random Walk o Biased Walk) ---
            idx = 0
            if self.external_flux is not None and self.external_flux.fluxStrength > 0:
                # CORREZIONE: Chiamata al metodo corretto di ParticleFlux
                weights = np.array([
                    self.external_flux.compute_external_flux_weights(step) for step in candidate_steps
                ])
                w_sum = weights.sum()
                if w_sum > 0:
                    idx = self.rng.choice(len(candidate_steps), p=weights/w_sum)
                else:
                    idx = self.rng.integers(len(candidate_steps))
            else:
                idx = self.rng.integers(len(candidate_steps))

            position += candidate_steps[idx]

            # --- 2. Controllo confini (Respawn se esce) ---
            (xmin, xmax), (ymin, ymax), (zmin, zmax) = outer_allowed_bounding_box
            if not (xmin <= position[0] <= xmax and
                    ymin <= position[1] <= ymax and
                    zmin <= position[2] <= zmax) or total_steps > max_steps:
                
                # Respawn
                position = initial_coordinate.copy()
                restarts += 1
                total_steps = 0
                continue

            # --- 3. Adesione (Sticking) ---
            # Controlla se siamo vicini a un vicino occupato
            neighbors = self.lattice.get_neighbors(*position)
            for nx, ny, nz in neighbors:
                if self.lattice.is_occupied(nx, ny, nz):
                    
                    # --- ANISOTROPIA CRISTALLINA (Miller) ---
                    # CORREZIONE: Controllo più robusto sulla lunghezza della lista
                    if len(self.lattice.preferred_axes) > 0:
                        
                        # Calcoliamo la probabilità strutturale basata sulla normale locale
                        a_s = self.lattice.compute_structural_probability(*position)
                        p_stick = min(1.0, a_s) # Normalizziamo a 1.0 per sicurezza
                        
                        # DEBUG per verificare se entra
                        # print(f"DEBUG: Attempting stick at {position}. Prob: {p_stick:.3f}")

                        if self.rng.random() > p_stick:
                            # REJECTION: La particella rimbalza.
                            # 'break' qui esce dal ciclo for dei vicini, 
                            # tornando al while True -> la particella fa un altro passo.
                            break 

                    # Se siamo qui, la particella si attacca
                    gid = self.lattice.get_group_id(nx, ny, nz)
                    self.lattice.occupy(*position, epoch=self.epoch, id=gid)
                    
                    # Registriamo le statistiche per debug/analisi
                    if self.lattice.collect_anisotropy_stats:
                         self.lattice.record_anisotropy_stats([position], self.epoch)

                    if self.verbose:
                        print(f"[DLAGrowth] Attached at {position} (Steps: {total_steps}, Restarts: {restarts})")
                    return

    def step(self):
        # Ottieni i box aggiornati in base alla crescita del cristallo
        generation_box = self.lattice.get_crystal_bounding_box(padding=self.generation_padding)
        outer_box = self.lattice.get_crystal_bounding_box(padding=self.outer_limit_padding)

        # Genera e muovi la particella
        start = self._generate_random_point_on_box(generation_box)
        self._particle_random_walk(start, outer_box)