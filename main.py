import Analysis as ANLS
import GUI as GUI
from classes.Lattice import Lattice
from classes.ParticleFlux import ParticleFlux
from classes.DLAGrowth import DLAGrowth
from classes.EDENGrowth import EDENGrowthKinetic
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
    EXTERNAL_FLUX:    Union[ParticleFlux, None]
    MILLER_INDICES:   Union[tuple, None]
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
        Flux Direction:   {self.EXTERNAL_FLUX}
        Miller Indices:   {self.MILLER_INDICES}
        Miller Strength:  {self.MILLER_STRENGTH}
        Miller Sharpness: {self.MILLER_SHARPNESS}
        Verbose:          {self.VERBOSE}
        Record:           {self.RECORD}
        """

def perform_EDEN_simulation(input: cusom_input):
    LATTICE = Lattice(input.NX, input.NY, input.NZ, input.VERBOSE)
    LATTICE.set_nucleation_seed(int(input.NX / 2), int(input.NY / 2), int(input.NZ / 2))
        
    model = EDENGrowthKinetic(lattice=LATTICE,
                              external_flux=input.EXTERNAL_FLUX,
                              three_dim=input.THREE_DIM,
                              verbose=input.VERBOSE)

    model.run(input.EPOCHS)
    
    GUI.plot_lattice(LATTICE, 
                     input.EPOCHS, 
                     title=input.TITLE, 
                     three_dim=input.THREE_DIM, 
                     out_dir=input.OUTPUT_DIR)
    

def perform_POLI_simulation(input: cusom_input):
    LATTICE = Lattice(input.NX, input.NY, input.NZ, input.VERBOSE)
    
    how_many_seeds = 0
    while how_many_seeds < 20:
        X = np.random.randint(0, input.NX)
        Y = np.random.randint(0, input.NY)
        Z = np.random.randint(0, input.NZ)
        
        if not LATTICE.is_occupied(X, Y, Z):
            LATTICE.set_nucleation_seed(X, Y, Z)
            how_many_seeds += 1

    model = EDENGrowthKinetic(lattice=LATTICE,
                              external_flux=input.EXTERNAL_FLUX,
                              three_dim=input.THREE_DIM,
                              verbose=input.VERBOSE)

    model.run(input.EPOCHS)
    
    GUI.plot_lattice(LATTICE, 
                     input.EPOCHS, 
                     title=input.TITLE, 
                     three_dim=input.THREE_DIM,
                     out_dir=input.OUTPUT_DIR, 
                     color_mode="id")
    GUI.plot_lattice(LATTICE, 
                     input.EPOCHS,
                     title=input.TITLE+'_boundaries', 
                     three_dim=input.THREE_DIM, 
                     out_dir=input.OUTPUT_DIR, 
                     color_mode="boundaries")
    

def perform_DLA_simulation(input: cusom_input):
    LATTICE = Lattice(input.NX, input.NY, input.NZ, input.VERBOSE)
    LATTICE.set_nucleation_seed(int(input.NX / 2), int(input.NY / 2), int(input.NZ / 2))

    if input.MILLER_INDICES is not None and len(input.MILLER_INDICES) == 3:
        if not all(v == 0 for v in input.MILLER_INDICES):
            h, k, l = input.MILLER_INDICES
            LATTICE.set_miller_anisotropy(h, k, l, 
                                          base_stick_prob=input.BASE_STICK_PROB,
                                          sticking_coefficient=input.MILLER_STRENGTH, 
                                          sharpness=input.MILLER_SHARPNESS, 
                                          selection_strength=input.MILLER_SELECTION)
            if input.RECORD:
                LATTICE.collect_anisotropy_stats = True

    print("Lattice initialized!")
    
    model = DLAGrowth(lattice=LATTICE,
                      generation_padding=1,
                      outer_limit_padding=3,
                      external_flux=input.EXTERNAL_FLUX,
                      three_dim=input.THREE_DIM,
                      verbose=input.VERBOSE)

    print("Model initialized!")
    model.run(input.EPOCHS)
    
    GUI.plot_lattice(LATTICE, 
                     input.EPOCHS, 
                     title=input.TITLE, 
                     three_dim=input.THREE_DIM, 
                     out_dir=input.OUTPUT_DIR)
    ANLS.fractal_dimention_analysis(LATTICE, 
                                    input.OUTPUT_DIR, 
                                    title=input.TITLE, 
                                    num_scales=25, 
                                    three_dim=input.THREE_DIM, 
                                    verbose=input.VERBOSE)
        

def perform_active_surface_simulation(input: cusom_input):
    LATTICE = Lattice(input.NX, input.NY, input.NZ, input.VERBOSE)
    for x in range(input.NX):
        for z in range(input.NZ):
            LATTICE.set_nucleation_seed(x, 0, z)

    # Default anisotropy: we aer simulating particles coming from +y direction
    flux = ParticleFlux(np.array([0, 1, 0]), 
                        input.EXTERNAL_FLUX.fluxStrength if input.EXTERNAL_FLUX.fluxStrength > 0.0 else 5.0, 
                        input.verbose)
    
    if input.MILLER_INDICES is not None and len(input.MILLER_INDICES) == 3:
        h, k, l = input.MILLER_INDICES
        LATTICE.set_miller_anisotropy(h, k, l, 
                                      base_stick_prob=input.BASE_STICK_PROB,
                                      sticking_coefficient=input.MILLER_STRENGTH, 
                                      sharpness=input.MILLER_SHARPNESS, 
                                      selection_strength=input.MILLER_SELECTION)
    
    model = DLAGrowth(LATTICE,
                      generation_padding=1,
                      outer_limit_padding=3,
                      external_flux=flux,
                      three_dim=input.THREE_DIM,
                      verbose=input.VERBOSE)

    model.run(input.EPOCHS)
    
    GUI.plot_lattice(LATTICE, 
                     input.EPOCHS, 
                     title=input.TITLE, 
                     three_dim=True, 
                     out_dir=input.OUTPUT_DIR)
    
    

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
    external_flux = ParticleFlux(parsed_inputs.external_flux,
                                 parsed_inputs.flux_strength,
                                 parsed_inputs.verbose) if parsed_inputs.external_flux is not None else None
    base_sticking_prob = parsed_inputs.base_stick
    miller_indices = parsed_inputs.miller
    miller_strength = parsed_inputs.anisotropy_coeff
    miller_sharpness = parsed_inputs.anisotropy_sharpness
    miller_selection_strength = parsed_inputs.anisotropy_selection
    
    simulation_input = cusom_input(NX=nx, NY=ny, NZ=nz, 
                                   EPOCHS=epochs, THREE_DIM=is_3D, 
                                   VERBOSE=verbose, RECORD=record,
                                   TITLE=title, OUTPUT_DIR=out_dir,
                                   EXTERNAL_FLUX=external_flux,
                                   BASE_STICK_PROB=base_sticking_prob,
                                   MILLER_INDICES=miller_indices, MILLER_STRENGTH=miller_strength, MILLER_SHARPNESS=miller_sharpness, MILLER_SELECTION=miller_selection_strength)
    
    if simulation_input.VERBOSE:
        print(simulation_input)

    if SIMULATION == 'EDEN':
        perform_EDEN_simulation(simulation_input)
        
    elif SIMULATION == 'POLI':
        perform_POLI_simulation(simulation_input)
        
    elif SIMULATION == 'DLA':
        perform_DLA_simulation(simulation_input)
        
    elif SIMULATION == 'SURFACE':
        perform_active_surface_simulation(simulation_input)

    else:
        print(f"***************************************************************************** \
                [MAIN LOOP] ERROR: simulation mode {SIMULATION} is not a valid option! \
                TERMINATING THE PROGRAM... \
                *****************************************************************************")