import EDEN_simulation as EDEN
import DLA_simulation as DLA
import Analysis as ANLS
import GUI as GUI
from Lattice import Lattice
from ArgParser import parse_inputs
import numpy as np
from dataclasses import dataclass
from typing import Union

@dataclass
class cusom_input:
    NX:               int
    NY:               int
    NZ:               int
    EPOCHS:           int
    THREE_DIM:        bool
    VERBOSE:          bool
    RECORD:           bool
    TITLE:            Union[str, None]
    OUTPUT_DIR:       Union[str, None]
    FLUX_DIRECTION:   Union[np.array, None]
    MILLER_INDICES:   Union[tuple, None]
    FLUX_STRENGTH:    float = 0.0
    BASE_STICK_PROB:  float = 0.01
    MILLER_STRENGTH:  float = 0.0
    MILLER_SHARPNESS: float = 4.0
    MILLER_SELECTION: float = 1.0
    # TODO: continue....
    
    def __str__(self):
        return f"""
        Simulation Settings:
        --------------------
        Size:             ({self.NX}, {self.NY}, {self.NZ})
        Epochs:           {self.EPOCHS}
        Dimensions:       {3 if self.THREE_DIM else 2}
        Title:            {self.TITLE}
        Output Dir:       {self.OUTPUT_DIR}
        Flux Direction:   {self.FLUX_DIRECTION}
        Flux Strength:    {self.FLUX_STRENGTH}
        Miller Indices:   {self.MILLER_INDICES}
        Miller Strength:  {self.MILLER_STRENGTH}
        Miller Sharpness: {self.MILLER_SHARPNESS}
        Verbose:          {self.VERBOSE}
        Record:           {self.RECORD}
        """

EDEN_OUTPUT_MESSAGES =  ["\nEDEN SIMULATION: COMPLETED SUCCESSFULLY!",
                        "\nEDEN SIMULATION: EARLY STOP. NO INITIAL NUCLEATION SEEDS FOUND!",
                        "\nEDEN SIMULATION: EARLY STOP. NO ACTIVE BORDER ON WHICH DEPOSIT THE PARTICLE!",
                        "\nEDEN SIMULATION: EARLY STOP."]
POLI_OUTPUT_MESSAGES  = ["\nEDEN SIMULATION: COMPLETED SUCCESSFULLY!",
                         "\nEDEN SIMULATION: EARLY STOP. NO INITIAL NUCLEATION SEEDS FOUND!",
                         "\nEDEN SIMULATION: EARLY STOP. NO ACTIVE BORDER ON WHICH DEPOSIT THE PARTICLE!",
                         "\nEDEN SIMULATION: EARLY STOP."]
 
def perform_EDEN_simulation(input: cusom_input):
    LATTICE = Lattice(input.NX, input.NY, input.NZ, input.VERBOSE)
    LATTICE.set_nucleation_seed(int(input.NX / 2), int(input.NY / 2), int(input.NZ / 2))
    
    if input.RECORD:
        LATTICE.collect_anisotropy_stats = True

    if input.FLUX_DIRECTION is not None and input.FLUX_STRENGTH > 0.0:
        LATTICE.set_external_flux(input.FLUX_DIRECTION, input.FLUX_STRENGTH)
        
    if input.MILLER_INDICES is not None and len(input.MILLER_INDICES) == 3:
        h, k, l = input.MILLER_INDICES
        LATTICE.set_miller_anisotropy(h, k, l, base_stick_prob=input.BASE_STICK_PROB, 
                                      sticking_coefficient=input.MILLER_STRENGTH, sharpness=input.MILLER_SHARPNESS, selection_strength=input.MILLER_SELECTION)
    
    output_code = EDEN.EDEN_simulation(LATTICE, input.EPOCHS, three_dim=input.THREE_DIM, verbose=input.VERBOSE, real_time_reference_point_correction=False)
    print(EDEN_OUTPUT_MESSAGES[output_code])
    
    GUI.plot_lattice(LATTICE, input.EPOCHS, title=input.TITLE, three_dim=input.THREE_DIM, out_dir=input.OUTPUT_DIR)
    

def perform_POLI_simulation(input: cusom_input):
    LATTICE = Lattice(input.NX, input.NY, input.NZ, input.VERBOSE)
    
    if input.RECORD:
        LATTICE.collect_anisotropy_stats = True
    
    how_many_seeds = 0
    while how_many_seeds < 20:
        X = np.random.randint(0, input.NX)
        Y = np.random.randint(0, input.NY)
        Z = np.random.randint(0, input.NZ)
        
        if not LATTICE.is_occupied(X, Y, Z):
            LATTICE.set_nucleation_seed(X, Y, Z)
            how_many_seeds += 1
            
    if input.FLUX_DIRECTION is not None and input.FLUX_STRENGTH > 0.0:
        LATTICE.set_external_flux(input.FLUX_DIRECTION, input.FLUX_STRENGTH)
        
    if input.MILLER_INDICES is not None and len(input.MILLER_INDICES) == 3:
        h, k, l = input.MILLER_INDICES
        LATTICE.set_miller_anisotropy(h, k, l, base_stick_prob=input.BASE_STICK_PROB,
                                      sticking_coefficient=input.MILLER_STRENGTH, sharpness=input.MILLER_SHARPNESS, selection_strength=input.MILLER_SELECTION)
            
    output_code = EDEN.EDEN_simulation(LATTICE, input.EPOCHS, three_dim=input.THREE_DIM, verbose=input.VERBOSE, real_time_reference_point_correction=False)
    print(POLI_OUTPUT_MESSAGES[output_code])
    
    GUI.plot_lattice(LATTICE, input.EPOCHS, title=input.TITLE, three_dim=input.THREE_DIM, out_dir=input.OUTPUT_DIR, color_mode="id")
    GUI.plot_lattice(LATTICE, input.EPOCHS, title=input.TITLE+'_boundaries', three_dim=input.THREE_DIM, out_dir=input.OUTPUT_DIR, color_mode="boundaries")
    

def perform_DLA_simulation(input: cusom_input):
    LATTICE = Lattice(input.NX, input.NY, input.NZ, input.VERBOSE)
    LATTICE.set_nucleation_seed(int(input.NX / 2), int(input.NY / 2), int(input.NZ / 2))
    
    if input.RECORD:
        LATTICE.collect_anisotropy_stats = True

    if input.FLUX_DIRECTION is not None and input.FLUX_STRENGTH > 0.0:
        LATTICE.set_external_flux(input.FLUX_DIRECTION, input.FLUX_STRENGTH)
        
    if input.MILLER_INDICES is not None and len(input.MILLER_INDICES) == 3:
        h, k, l = input.MILLER_INDICES
        LATTICE.set_miller_anisotropy(h, k, l, base_stick_prob=input.BASE_STICK_PROB,
                                      sticking_coefficient=input.MILLER_STRENGTH, sharpness=input.MILLER_SHARPNESS, selection_strength=input.MILLER_SELECTION)
    
    s_mean, s_std, r_mean, r_std = DLA.DLA_simulation(LATTICE, input.EPOCHS, 1, 3, three_dim=input.THREE_DIM, verbose=input.VERBOSE)
    print(f"\nDLA SIMULATION COMPLETED!\n \
          Statistics about the random walk:\n \
          \t* Mean number of steps in the random walk: {s_mean} +/- {s_std}\n \
          \t* Mean number of restarts during random walk: {r_mean} +/- {r_std}")
    
    GUI.plot_lattice(LATTICE, input.EPOCHS, title=input.TITLE, three_dim=input.THREE_DIM, out_dir=input.OUTPUT_DIR)
    
    if input.OUTPUT_DIR is not None:
        ANLS.fractal_dimention_analysis(LATTICE, input.OUTPUT_DIR, title=input.TITLE, num_scales=25, three_dim=input.THREE_DIM, verbose=input.VERBOSE)
        
        
def perform_active_surface_simulation(input: cusom_input):
    LATTICE = Lattice(input.NX, input.NY, input.NZ, input.VERBOSE)
    for x in range(input.NX):
        for z in range(input.NZ):
            LATTICE.set_nucleation_seed(x, 0, z)
            
    if input.RECORD:
        LATTICE.collect_anisotropy_stats = True

    # Default anisotropy: we aer simulating particles coming from +y direction
    LATTICE.set_external_flux(np.array([0,1,0]), input.FLUX_STRENGTH)
    
    if input.MILLER_INDICES is not None and len(input.MILLER_INDICES) == 3:
        h, k, l = input.MILLER_INDICES
        LATTICE.set_miller_anisotropy(h, k, l, base_stick_prob=input.BASE_STICK_PROB,
                                      sticking_coefficient=input.MILLER_STRENGTH, sharpness=input.MILLER_SHARPNESS, selection_strength=input.MILLER_SELECTION)
    
            
    s_mean, s_std, r_mean, r_std = DLA.DLA_simulation(LATTICE, input.EPOCHS, 1, 3, three_dim=True, verbose=input.VERBOSE)
    print(f"\nDLA SIMULATION FOR ACTIVE SURFACE COMPLETED!\n \
          Statistics about the random walk:\n \
          \t* Mean number of steps in the random walk: {s_mean} +/- {s_std}\n \
          \t* Mean number of restarts during random walk: {r_mean} +/- {r_std}")
    
    GUI.plot_lattice(LATTICE, input.EPOCHS, title=input.TITLE, three_dim=True, out_dir=input.OUTPUT_DIR)
    
    if input.OUTPUT_DIR is not None:
        ANLS.distance_from_active_surface(LATTICE, input.OUTPUT_DIR, input.EPOCHS, verbose=input.VERBOSE)
    
    

if __name__ == '__main__':
    parsed_inputs = parse_inputs()
    
    SIMULATION = parsed_inputs.simulation
    epochs = parsed_inputs.epochs
    nx, ny, nz = parsed_inputs.size
    is_3D = not parsed_inputs.two_dim
    title = parsed_inputs.title
    verbose = parsed_inputs.verbose
    record = parsed_inputs.record
    out_dir = None if parsed_inputs.output == "" else parsed_inputs.output
    flux_direction = parsed_inputs.external_flux
    flux_strength = parsed_inputs.flux_strength
    base_sticking_prob = parsed_inputs.base_stick
    miller_indices = parsed_inputs.miller
    miller_strength = parsed_inputs.anisotropy_coeff
    miller_sharpness = parsed_inputs.anisotropy_sharpness
    miller_selection_strength = parsed_inputs.anisotropy_selection
    
    simulation_input = cusom_input(NX=nx, NY=ny, NZ=nz, 
                                   EPOCHS=epochs, THREE_DIM=is_3D, 
                                   VERBOSE=verbose, RECORD=record,
                                   TITLE=title, OUTPUT_DIR=out_dir,
                                   FLUX_DIRECTION=flux_direction, FLUX_STRENGTH=flux_strength,
                                   BASE_STICK_PROB=base_sticking_prob,
                                   MILLER_INDICES=miller_indices, MILLER_STRENGTH=miller_strength, MILLER_SHARPNESS=miller_sharpness, MILLER_SELECTION=miller_selection_strength)
    
    if SIMULATION == 'EDEN':
        perform_EDEN_simulation(simulation_input)
        
    elif SIMULATION == 'POLI':
        perform_POLI_simulation(simulation_input)
        
    elif SIMULATION == 'DLA':
        perform_DLA_simulation(simulation_input)
        
    elif SIMULATION == 'SURFACE':
        perform_active_surface_simulation(simulation_input)