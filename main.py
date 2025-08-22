import EDEN_simulation as EDEN
import DLA_simulation as DLA
import Analysis as ANLS
import GUI as GUI
from Lattice import Lattice
from ArgParser import parse_inputs

def perform_EDEN_simulation(NX: int, NY: int, NZ: int, N_EPOCHS: int, three_dim: bool, verbose: bool, title: str):
    LATTICE = Lattice(NX, NY, NZ)
    LATTICE.set_nucleation_seed(int(NX / 2), int(NY / 2), int(NZ / 2))
    
    output_code = EDEN.EDEN_simulation(LATTICE, N_EPOCHS, three_dim=three_dim, verbose=verbose)
    output_messages = ["\nEDEN SIMULATION: COMPLETED SUCCESSFULLY!",
                       "\nEDEN SIMULATION: EARLY STOP. NO INITIAL NUCLEATION SEEDS FOUND!",
                       "\nEDEN SIMULATION: EARLY STOP. NO ACTIVE BORDER ON WHICH DEPOSIT THE PARTICLE!",
                       "\nEDEN SIMULATION: EARLY STOP."]
    print(output_messages[output_code])
    
    # TODO: optional analysis part
    GUI.plot_lattice(LATTICE, N_EPOCHS, title=title, three_dim=three_dim)

def perform_DLA_simulation(NX: int, NY: int, NZ: int, N_EPOCHS: int, three_dim: bool, verbose: bool, title: str):
    LATTICE = Lattice(NX, NY, NZ)
    LATTICE.set_nucleation_seed(int(NX / 2), int(NY / 2), int(NZ / 2))
    
    s_mean, s_std, r_mean, r_std = DLA.DLA_simulation(LATTICE, N_EPOCHS, 1, 3, three_dim=three_dim, verbose=verbose)
    print(f"\nDLA SIMULATION COMPLETED!\n \
          Statistics about the random walk:\n \
          \t* Mean number of steps in the random walk: {s_mean} +/- {s_std}\n \
          \t* Mean number of restarts during random walk: {r_mean} +/- {r_std}")
    
    # TODO: optional analysis part
    GUI.plot_lattice(LATTICE, N_EPOCHS, title=title, three_dim=three_dim)

def perform_active_surface_simulation(NX: int, NY: int, NZ: int, N_EPOCHS: int, verbose: bool, title: str):
    LATTICE = Lattice(NX, NY, NZ)
    for x in range(NX):
        for z in range(NZ):
            LATTICE.set_nucleation_seed(x, 0, z)
            
    s_mean, s_std, r_mean, r_std = DLA.DLA_simulation(LATTICE, N_EPOCHS, 1, 3, three_dim=True, verbose=verbose)
    print(f"\nDLA SIMULATION FOR ACTIVE SURFACE COMPLETED!\n \
          Statistics about the random walk:\n \
          \t* Mean number of steps in the random walk: {s_mean} +/- {s_std}\n \
          \t* Mean number of restarts during random walk: {r_mean} +/- {r_std}")
    
    # TODO: optional analysis specific for this simulation????
    GUI.plot_lattice(LATTICE, N_EPOCHS, title=title, three_dim=True)

if __name__ == '__main__':
    parsed_inputs = parse_inputs()
    EPOCHS = parsed_inputs.epochs
    NX, NY, NZ = parsed_inputs.size
    IS_3D = not parsed_inputs.two_dim
    TITLE = parsed_inputs.title
    SIMULATION = parsed_inputs.simulation
    VERBOSE = parsed_inputs.verbose
    
    if SIMULATION == 'EDEN':
        perform_EDEN_simulation(NX, NY, NZ, EPOCHS, IS_3D, VERBOSE, TITLE)
        
    elif SIMULATION == 'DLA':
        perform_DLA_simulation(NX, NY, NZ, EPOCHS, IS_3D, VERBOSE, TITLE)
        
    elif SIMULATION == 'SURFACE':
        perform_active_surface_simulation(NX, NY, NZ, EPOCHS, VERBOSE, TITLE)