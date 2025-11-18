import EDEN_simulation as EDEN
import DLA_simulation as DLA
import Analysis as ANLS
import GUI as GUI
from Lattice import Lattice
from ArgParser import parse_inputs
import numpy as np

def perform_EDEN_simulation(NX: int, NY: int, NZ: int, N_EPOCHS: int, three_dim: bool, verbose: bool, title: str, out_dir: str = None,
                           anisotropy_directions: np.array = None, anisotropy_strength: float = 0.0):
    LATTICE = Lattice(NX, NY, NZ)
    LATTICE.set_nucleation_seed(int(NX / 2), int(NY / 2), int(NZ / 2))

    if anisotropy_directions is not None and anisotropy_strength > 0.0:
        LATTICE.setAnisotropy(anisotropy_directions, anisotropy_strength)
    
    output_code = EDEN.EDEN_simulation(LATTICE, N_EPOCHS, three_dim=three_dim, verbose=verbose, real_time_reference_point_correction=False)
    output_messages = ["\nEDEN SIMULATION: COMPLETED SUCCESSFULLY!",
                       "\nEDEN SIMULATION: EARLY STOP. NO INITIAL NUCLEATION SEEDS FOUND!",
                       "\nEDEN SIMULATION: EARLY STOP. NO ACTIVE BORDER ON WHICH DEPOSIT THE PARTICLE!",
                       "\nEDEN SIMULATION: EARLY STOP."]
    print(output_messages[output_code])
    
    GUI.plot_lattice(LATTICE, N_EPOCHS, title=title, three_dim=three_dim, out_dir=out_dir)
    
    

def perform_DLA_simulation(NX: int, NY: int, NZ: int, N_EPOCHS: int, three_dim: bool, verbose: bool, title: str, out_dir: str = None,
                          anisotropy_directions=None, anisotropy_strength: float = 0.0):
    LATTICE = Lattice(NX, NY, NZ)
    LATTICE.set_nucleation_seed(int(NX / 2), int(NY / 2), int(NZ / 2))

    if anisotropy_directions is not None and anisotropy_stregth > 0.0:
        LATTICE.setAnisotropy(anisotropy_directions, anisotropy_strength)
    
    s_mean, s_std, r_mean, r_std = DLA.DLA_simulation(LATTICE, N_EPOCHS, 1, 3, three_dim=three_dim, verbose=verbose)
    print(f"\nDLA SIMULATION COMPLETED!\n \
          Statistics about the random walk:\n \
          \t* Mean number of steps in the random walk: {s_mean} +/- {s_std}\n \
          \t* Mean number of restarts during random walk: {r_mean} +/- {r_std}")
    
    GUI.plot_lattice(LATTICE, N_EPOCHS, title=title, three_dim=three_dim, out_dir=out_dir)
    
    if out_dir is not None:
        ANLS.fractal_dimention_analysis(LATTICE, out_dir, num_scales=25, three_dim=three_dim, verbose=verbose)
        
        

def perform_active_surface_simulation(NX: int, NY: int, NZ: int, N_EPOCHS: int, verbose: bool, title: str, out_dir: str = None, anisotropy_strength: float = 0.0):
    LATTICE = Lattice(NX, NY, NZ)
    for x in range(NX):
        for z in range(NZ):
            LATTICE.set_nucleation_seed(x, 0, z)

    # Default anisotropy: we aer simulating particles coming from +y direction
    LATTICE.setAnisotropy(np.array([0,1,0]), anisotropy_strength)
            
    s_mean, s_std, r_mean, r_std = DLA.DLA_simulation(LATTICE, N_EPOCHS, 1, 3, three_dim=True, verbose=verbose)
    print(f"\nDLA SIMULATION FOR ACTIVE SURFACE COMPLETED!\n \
          Statistics about the random walk:\n \
          \t* Mean number of steps in the random walk: {s_mean} +/- {s_std}\n \
          \t* Mean number of restarts during random walk: {r_mean} +/- {r_std}")
    
    GUI.plot_lattice(LATTICE, N_EPOCHS, title=title, three_dim=True, out_dir=out_dir)
    
    if out_dir is not None:
        ANLS.distance_from_active_surface(LATTICE, out_dir, N_EPOCHS, verbose=verbose)
    
    

if __name__ == '__main__':
    parsed_inputs = parse_inputs()
    EPOCHS = parsed_inputs.epochs
    NX, NY, NZ = parsed_inputs.size
    IS_3D = not parsed_inputs.two_dim
    TITLE = parsed_inputs.title
    SIMULATION = parsed_inputs.simulation
    VERBOSE = parsed_inputs.verbose
    OUTPUT_DIR = None if parsed_inputs.output == "" else parsed_inputs.output
    ANISOTROPIC_DIRECTIONS = parsed_inputs.anisotropy_directions
    ANISOTROPIC_STRENGTH = parsed_inputs.anisotropy_strength
    
    if SIMULATION == 'EDEN':
        perform_EDEN_simulation(NX, NY, NZ, EPOCHS, IS_3D, VERBOSE, TITLE, OUTPUT_DIR, ANISOTROPIC_DIRECTIONS, ANISOTROPIC_STRENGTH)
        
    elif SIMULATION == 'DLA':
        perform_DLA_simulation(NX, NY, NZ, EPOCHS, IS_3D, VERBOSE, TITLE, OUTPUT_DIR, ANISOTROPIC_DIRECTIONS, ANISOTROPIC_STRENGTH)
        
    elif SIMULATION == 'SURFACE':
        perform_active_surface_simulation(NX, NY, NZ, EPOCHS, VERBOSE, TITLE, OUTPUT_DIR, ANISOTROPIC_DIRECTIONS, ANISOTROPIC_STRENGTH)
