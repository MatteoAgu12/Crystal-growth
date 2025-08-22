import EDEN_simulation as EDEN
import DLA_simulation as DLA
import Analysis as ANLS
import GUI as GUI
from Lattice import Lattice
from ArgParser import parse_inputs

def perform_EDEN_simulation(NX: int, NY: int, NZ: int, N_EPOCHS: int, three_dim: bool, title: str):
    LATTICE = Lattice(NX, NY, NZ)
    LATTICE.set_nucleation_seed(int(NX / 2), int(NY / 2), int(NZ / 2))
    
    output_code = EDEN.EDEN_simulation(LATTICE, N_EPOCHS, three_dim=three_dim)
    output_messages = ["EDEN SIMULATION: COMPLETED SUCCESSFULLY!",
                       "EDEN SIMULATION: EARLY STOP. NO INITIAL NUCLEATION SEEDS FOUND!",
                       "EDEN SIMULATION: EARLY STOP. NO ACTIVE BORDER ON WHICH DEPOSIT THE PARTICLE!",
                       "EDEN SIMULATION: EARLY STOP."]
    print(output_messages[output_code])
    
    # TODO: optional analysis part
    GUI.plot_lattice(LATTICE, N_EPOCHS, title=title, three_dim=three_dim)

def perform_DLA_simulation(NX: int, NY: int, NZ: int, N_EPOCHS: int, three_dim: bool, title: str):
    LATTICE = Lattice(NX, NY, NZ)
    LATTICE.set_nucleation_seed(int(NX / 2), int(NY / 2), int(NZ / 2))
    
    s_mean, s_std, r_mean, r_std = DLA.DLA_simulation(LATTICE, N_EPOCHS, 1, 3, three_dim=three_dim)
    print(f"DLA SIMULATION COMPLETED!\n \
          Statistics about the random walk:\n \
          \t* Mean number of steps in the random walk: {s_mean} +/- {s_std}\n \
          \t* Mean number of restarts during random walk: {r_mean} +/- {r_std}")
    
    # TODO: optional analysis part
    GUI.plot_lattice(LATTICE, N_EPOCHS, title=title, three_dim=three_dim)

def perform_active_surface_simulation():
    pass

if __name__ == '__main__':
    print(parse_inputs())