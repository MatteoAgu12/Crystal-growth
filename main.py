import numpy as np
from dataclasses import dataclass
from typing import Union

from classes.KineticLattice import KineticLattice
from classes.PhaseFieldLattice import PhaseFieldLattice
from classes.ParticleFlux import ParticleFlux
from classes.DLAGrowth import DLAGrowth
from classes.EDENGrowth import EDENGrowth
from classes.KobayashiGrowth import KobayashiGrowth
from classes.StefanGrowth import StefanGrowth

from utils.ArgParser import parse_inputs
import utils.Analysis as ANLS
import utils.GUI as GUI

@dataclass
class custom_input:
    NX:               int
    NY:               int
    NZ:               int
    SEEDS:            int
    EPOCHS:           int
    THREE_DIM:        bool
    VERBOSE:          bool
    TITLE:            str
    OUTPUT_DIR:       str
    EXTERNAL_FLUX:    ParticleFlux

    # KOBA only
    INTERFACE_THR:    float
    EPSILON:          float
    DELTA:            float
    N_FOLDS:          float
    ALPHA:            float
    U_EQ:             float
    U_INFTY:          float
    LATENT_COEF:      float
    DIFFUSIVITY:      float
    MOBILITY:         float
    SUPERSATURATION:  float
    TIME_STEP:        float
    # TODO: continue....
    
    def __str__(self):
        # TODO: aggiorna il print
        return f"""
        Simulation Settings:
        --------------------
        Size:             ({self.NX}, {self.NY}, {self.NZ})
        Seeds:            {self.SEEDS}
        Epochs:           {self.EPOCHS}
        Dimensions:       {3 if self.THREE_DIM else 2}
        Title:            {self.TITLE}
        Output Dir:       {self.OUTPUT_DIR}
        Flux Direction:   {self.EXTERNAL_FLUX}
        Verbose:          {self.VERBOSE}
        """


def perform_EDEN_simulation(input: custom_input):
    LATTICE = KineticLattice(input.NX, input.NY, input.NZ, input.VERBOSE)

    if input.SEEDS == 1:
        LATTICE.set_nucleation_seed(int(input.NX / 2), int(input.NY / 2), int(input.NZ / 2))
    else:
        how_many_seeds = 0
        while how_many_seeds < input.SEEDS:
            X = np.random.randint(0, input.NX)
            Y = np.random.randint(0, input.NY)
            Z = np.random.randint(0, input.NZ)

            if not LATTICE.is_occupied(X, Y, Z):
                LATTICE.set_nucleation_seed(X, Y, Z)
                how_many_seeds += 1
        
    model = EDENGrowth(lattice=LATTICE,
                              external_flux=input.EXTERNAL_FLUX,
                              three_dim=input.THREE_DIM,
                              verbose=input.VERBOSE)

    model.run(input.EPOCHS)
    
    GUI.plot_kinetic_lattice(LATTICE, 
                     input.EPOCHS, 
                     title=input.TITLE, 
                     three_dim=input.THREE_DIM, 
                     out_dir=input.OUTPUT_DIR)
    GUI.plot_kinetic_lattice(LATTICE, 
                     input.EPOCHS, 
                     title=input.TITLE+"_id", 
                     three_dim=input.THREE_DIM, 
                     out_dir=input.OUTPUT_DIR,
                     color_mode="id")
    GUI.plot_kinetic_lattice(LATTICE, 
                     input.EPOCHS,
                     title=input.TITLE+'_boundaries', 
                     three_dim=input.THREE_DIM, 
                     out_dir=input.OUTPUT_DIR, 
                     color_mode="boundaries")
       

def perform_DLA_simulation(input: custom_input):
    LATTICE = KineticLattice(input.NX, input.NY, input.NZ, input.VERBOSE)
    
    if input.SEEDS == 1:
        LATTICE.set_nucleation_seed(int(input.NX / 2), int(input.NY / 2), int(input.NZ / 2))
    else:
        how_many_seeds = 0
        while how_many_seeds < input.SEEDS:
            X = np.random.randint(0, input.NX)
            Y = np.random.randint(0, input.NY)
            Z = np.random.randint(0, input.NZ)

            if not LATTICE.is_occupied(X, Y, Z):
                LATTICE.set_nucleation_seed(X, Y, Z)
                how_many_seeds += 1
    
    model = DLAGrowth(lattice=LATTICE,
                      generation_padding=1,
                      outer_limit_padding=3,
                      external_flux=input.EXTERNAL_FLUX,
                      three_dim=input.THREE_DIM,
                      verbose=input.VERBOSE)

    model.run(input.EPOCHS)

    if input.SEEDS == 1:
        ANLS.fractal_dimention_analysis(LATTICE, 
                                        input.OUTPUT_DIR, 
                                        title=input.TITLE, 
                                        num_scales=25, 
                                        three_dim=input.THREE_DIM, 
                                        verbose=input.VERBOSE)
    
    GUI.plot_kinetic_lattice(LATTICE, 
                     input.EPOCHS, 
                     title=input.TITLE, 
                     three_dim=input.THREE_DIM, 
                     out_dir=input.OUTPUT_DIR)
    GUI.plot_kinetic_lattice(LATTICE, 
                     input.EPOCHS, 
                     title=input.TITLE+"_id", 
                     three_dim=input.THREE_DIM, 
                     out_dir=input.OUTPUT_DIR,
                     color_mode="id")
    GUI.plot_kinetic_lattice(LATTICE, 
                     input.EPOCHS,
                     title=input.TITLE+'_boundaries', 
                     three_dim=input.THREE_DIM, 
                     out_dir=input.OUTPUT_DIR, 
                     color_mode="boundaries")
        

def perform_KOBAYASHI_simulation(input: custom_input):
    LATTICE = PhaseFieldLattice(input.NX, input.NY, input.NZ, input.INTERFACE_THR, input.VERBOSE)

    if input.SEEDS == 1:
        LATTICE.set_nucleation_seed(int(input.NX / 2), int(input.NY / 2), int(input.NZ / 2))
    else:
        how_many_seeds = 0
        while how_many_seeds < input.SEEDS:
            X = np.random.randint(0, input.NX)
            Y = np.random.randint(0, input.NY)
            Z = np.random.randint(0, input.NZ)

            if not LATTICE.is_occupied(X, Y, Z):
                LATTICE.set_nucleation_seed(X, Y, Z)
                how_many_seeds += 1

    model = KobayashiGrowth(LATTICE,
                            epsilon0=input.EPSILON,
                            delta=input.DELTA,
                            n_folds=input.N_FOLDS,
                            mobility=input.MOBILITY,
                            supersaturation=input.SUPERSATURATION,
                            dt=input.TIME_STEP,
                            external_flux=None,
                            three_dim=input.THREE_DIM,
                            verbose=input.VERBOSE)

    model.run(input.EPOCHS)
    
    GUI.plot_phase_field_simulation(LATTICE,
                                    out_dir=input.OUTPUT_DIR,
                                    field_name="phi",
                                    color_field_name="phi",
                                    title=input.TITLE,
                                    three_dim=input.THREE_DIM)
    GUI.plot_phase_field_simulation(LATTICE,
                                    out_dir=input.OUTPUT_DIR,
                                    field_name="history",
                                    color_field_name="history",
                                    title=input.TITLE,
                                    three_dim=input.THREE_DIM)
    GUI.plot_phase_field_simulation(LATTICE,
                                    out_dir=input.OUTPUT_DIR,
                                    field_name="curvature",
                                    color_field_name="curvature",
                                    title=input.TITLE,
                                    three_dim=input.THREE_DIM)

    GUI.plot_kinetic_lattice(LATTICE, 
                     input.EPOCHS, 
                     title=input.TITLE+"_id", 
                     three_dim=input.THREE_DIM, 
                     out_dir=input.OUTPUT_DIR,
                     color_mode="id")
    GUI.plot_kinetic_lattice(LATTICE, 
                     input.EPOCHS,
                     title=input.TITLE+'_boundaries', 
                     three_dim=input.THREE_DIM, 
                     out_dir=input.OUTPUT_DIR, 
                     color_mode="boundaries")


def perform_STEFAN_simulation(input: custom_input):
    LATTICE = PhaseFieldLattice(input.NX, input.NY, input.NZ, input.INTERFACE_THR, input.VERBOSE)
    LATTICE.u[:,:,:] = input.U_INFTY

    if input.SEEDS == 1:
        LATTICE.set_nucleation_seed(int(input.NX / 2), int(input.NY / 2), int(input.NZ / 2))
        phi = LATTICE.phi[:,:,:]
        mask = (phi > 0.1) & (phi < 0.9)
        phi[mask] += 0.02 * np.random.default_rng(0).normal(size=phi.shape)[mask]
        LATTICE.phi[:,:,:] = np.clip(phi, 0.0, 1.0)

    else:
        how_many_seeds = 0
        while how_many_seeds < input.SEEDS:
            X = np.random.randint(0, input.NX)
            Y = np.random.randint(0, input.NY)
            Z = np.random.randint(0, input.NZ)

            if not LATTICE.is_occupied(X, Y, Z):
                LATTICE.set_nucleation_seed(X, Y, Z)
                phi = LATTICE.phi[:,:,:]
                mask = (phi > 0.1) & (phi < 0.9)
                phi[mask] += 0.02 * np.random.default_rng(0).normal(size=phi.shape)[mask]
                LATTICE.phi[:,:,:] = np.clip(phi, 0.0, 1.0)
                how_many_seeds += 1

    model = StefanGrowth(LATTICE,
                            epsilon0=input.EPSILON,
                            delta=input.DELTA,
                            n_folds=input.N_FOLDS,
                            mobility=input.MOBILITY,
                            diffusivity=input.DIFFUSIVITY,
                            latent_coeff=input.LATENT_COEF,
                            alpha=input.ALPHA,
                            u_eq=input.U_EQ,
                            u_infty=input.U_INFTY,
                            enforce_dirichlet_u=True, # TODO: qui decidere se tenere o no
                            dt=input.TIME_STEP,
                            external_flux=None,
                            three_dim=input.THREE_DIM,
                            verbose=input.VERBOSE)

    # z = input.NZ // 2
    # print("init u[min,max] =", LATTICE.u[:, :, z].min(), LATTICE.u[:, :, z].max())
    # print("init phi[min,max] =", LATTICE.phi[:, :, z].min(), LATTICE.phi[:, :, z].max())
    # print("alpha =", input.ALPHA, "L =", input.LATENT_COEF, "u_infty =", input.U_INFTY)
    # return

    model.run(input.EPOCHS)
    
    GUI.plot_phase_field_simulation(LATTICE,
                                    out_dir=input.OUTPUT_DIR,
                                    field_name = "phi",
                                    color_field_name="phi",
                                    title=input.TITLE,
                                    three_dim=input.THREE_DIM)
    GUI.plot_phase_field_simulation(LATTICE,
                                    out_dir=input.OUTPUT_DIR,
                                    field_name="u",
                                    color_field_name="u",
                                    title=input.TITLE,
                                    three_dim=input.THREE_DIM)
    GUI.plot_phase_field_simulation(LATTICE,
                                    out_dir=input.OUTPUT_DIR,
                                    field_name="history",
                                    color_field_name="history",
                                    title=input.TITLE,
                                    three_dim=input.THREE_DIM)
    GUI.plot_phase_field_simulation(LATTICE,
                                    out_dir=input.OUTPUT_DIR,
                                    field_name="curvature",
                                    color_field_name="curvature",
                                    title=input.TITLE,
                                    three_dim=input.THREE_DIM)

    # GUI.plot_kinetic_lattice(LATTICE, 
    #                  input.EPOCHS, 
    #                  title=input.TITLE+"_id", 
    #                  three_dim=input.THREE_DIM, 
    #                  out_dir=input.OUTPUT_DIR,
    #                  color_mode="id")
    # GUI.plot_kinetic_lattice(LATTICE, 
    #                  input.EPOCHS,
    #                  title=input.TITLE+'_boundaries', 
    #                  three_dim=input.THREE_DIM, 
    #                  out_dir=input.OUTPUT_DIR, 
    #                  color_mode="boundaries")

# TODO: rimuovere
def perform_active_surface_simulation(input: custom_input):
    LATTICE = KineticLattice(input.NX, input.NY, input.NZ, input.VERBOSE)
    for x in range(input.NX):
        for z in range(input.NZ):
            LATTICE.set_nucleation_seed(x, 0, z)

    # Default anisotropy: we aer simulating particles coming from +y direction
    flux = ParticleFlux(np.array([0, 1, 0]), 
                        input.EXTERNAL_FLUX.fluxStrength if input.EXTERNAL_FLUX.fluxStrength > 0.0 else 5.0, 
                        input.verbose)
    
    model = DLAGrowth(LATTICE,
                      generation_padding=1,
                      outer_limit_padding=3,
                      external_flux=flux,
                      three_dim=input.THREE_DIM,
                      verbose=input.VERBOSE)

    model.run(input.EPOCHS)
    
    GUI.plot_kinetic_lattice(LATTICE, 
                     input.EPOCHS, 
                     title=input.TITLE, 
                     three_dim=True, 
                     out_dir=input.OUTPUT_DIR)
    
    

if __name__ == '__main__':
    # ================================================================
    # Reading and collecting the inuts from the user
    # ================================================================
    parsed_inputs = parse_inputs()
    
    SIMULATION    = parsed_inputs.simulation
    epochs        = parsed_inputs.epochs
    nx, ny, nz    = parsed_inputs.size
    seeds         = parsed_inputs.seeds
    is_3D         = not parsed_inputs.two_dim
    title         = parsed_inputs.title
    verbose       = parsed_inputs.verbose
    out_dir       = None if parsed_inputs.output == "" else parsed_inputs.output
    external_flux = ParticleFlux(parsed_inputs.external_flux,
                                 parsed_inputs.flux_strength,
                                 parsed_inputs.verbose) if parsed_inputs.external_flux is not None else None

    # Phase Field only
    interface_thr   = parsed_inputs.interface_thr
    epsilon0        = parsed_inputs.epsilon0
    delta           = parsed_inputs.delta
    n_folds         = parsed_inputs.n_folds
    alpha           = parsed_inputs.alpha
    u_eq            = parsed_inputs.u_equilibrium
    u_infinity      = parsed_inputs.u_infinity
    latent_coef     = parsed_inputs.latent_coef
    diffusivity     = parsed_inputs.diffusivity
    mobility        = parsed_inputs.mobility
    supersaturation = parsed_inputs.supersaturation
    time_step      = parsed_inputs.dt
    
    # ================================================================
    # Creating the input objects
    # ================================================================
    simulation_input = custom_input(NX=nx, NY=ny, NZ=nz,
                                   SEEDS=seeds,
                                   EPOCHS=epochs, 
                                   THREE_DIM=is_3D, 
                                   VERBOSE=verbose,
                                   TITLE=title, OUTPUT_DIR=out_dir,
                                   EXTERNAL_FLUX=external_flux,
                                   INTERFACE_THR=interface_thr,
                                   EPSILON=epsilon0,
                                   DELTA=delta,
                                   N_FOLDS=n_folds,
                                   ALPHA=alpha,
                                   U_EQ=u_eq,
                                   U_INFTY=u_infinity,
                                   LATENT_COEF=latent_coef,
                                   MOBILITY=mobility,
                                   DIFFUSIVITY=diffusivity,
                                   SUPERSATURATION=supersaturation,
                                   TIME_STEP=time_step)
    
    print(simulation_input)

    # ================================================================
    # Run the desired simulation
    # ================================================================
    if SIMULATION == 'EDEN':
        perform_EDEN_simulation(simulation_input)
        
    elif SIMULATION == 'DLA':
        perform_DLA_simulation(simulation_input)

    elif SIMULATION == 'KOBAYASHI':
        perform_KOBAYASHI_simulation(simulation_input)

    elif SIMULATION == 'STEFAN':
        perform_STEFAN_simulation(simulation_input)

    # elif SIMULATION == 'SURFACE':
    #     perform_active_surface_simulation(simulation_input)

    else:
        print(f"***************************************************************************** \
                [MAIN LOOP] ERROR: simulation mode {SIMULATION} is not a valid option! \
                TERMINATING THE PROGRAM... \
                *****************************************************************************")