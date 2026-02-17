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

import utils.Analysis as ANLS
import utils.GUI as GUI

import logging 
logger = logging.getLogger("growthsim")

@dataclass
class custom_input:
    SIMULATION:       str
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
    GAMMA:            float    
    DIFFUSIVITY:      float
    MOBILITY:         float
    SUPERSATURATION:  float
    TIME_STEP:        float
    
    def __str__(self):
        if self.SIMULATION in ['EDEN', 'DLA']:
            return f"""
            ===============================
            {self.SIMULATION} Simulation Settings:
            ===============================
            GLOBAL PARAMETERS:
            Size:             ({self.NX}, {self.NY}, {self.NZ})
            Seeds:            {self.SEEDS}
            Epochs:           {self.EPOCHS}
            Dimensions:       {3 if self.THREE_DIM else 2}
            Title:            {self.TITLE}
            Output Dir:       {self.OUTPUT_DIR}
            Flux Direction:   {self.EXTERNAL_FLUX}
            Verbose:          {self.VERBOSE}
            --------------------------------
            """
        else:
            return f"""
            ===============================
            {self.SIMULATION} Simulation Settings:
            ===============================
            GLOBAL PARAMETERS:
            Size:             ({self.NX}, {self.NY}, {self.NZ})
            Seeds:            {self.SEEDS}
            Epochs:           {self.EPOCHS}
            Dimensions:       {3 if self.THREE_DIM else 2}
            Title:            {self.TITLE}
            Output Dir:       {self.OUTPUT_DIR}
            Flux Direction:   {self.EXTERNAL_FLUX}
            Verbose:          {self.VERBOSE}
            --------------------------------

            Phase Field Parameters:
            Interface Thr:    {self.INTERFACE_THR}
            Epsilon:          {self.EPSILON}
            Delta:            {self.DELTA}
            N Folds:          {self.N_FOLDS}
            Alpha:            {self.ALPHA}
            U_eq:             {self.U_EQ}
            U_infty:          {self.U_INFTY}
            Latent Coef:      {self.LATENT_COEF}
            Gamma:            {self.GAMMA}
            Diffusivity:      {self.DIFFUSIVITY}
            Mobility:         {self.MOBILITY}
            Supersaturation:  {self.SUPERSATURATION}
            Time Step:        {self.TIME_STEP}
            --------------------------------
            """

def _init_kinetic_lattice(input: custom_input) -> KineticLattice:
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

    return LATTICE

def _init_phase_field_lattice(input: custom_input) -> PhaseFieldLattice:
    LATTICE = PhaseFieldLattice(input.NX, input.NY, 1, input.INTERFACE_THR, input.VERBOSE)
    LATTICE.u[:,:,:] = input.U_INFTY
    phi = LATTICE.phi[:,:,:]

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
    
    return LATTICE


def perform_EDEN_simulation(input: custom_input):
    LATTICE = _init_kinetic_lattice(input)

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

    if input.SEEDS != 1:
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
    LATTICE = _init_kinetic_lattice(input)
    
    model = DLAGrowth(lattice=LATTICE,
                      generation_padding=1,
                      outer_limit_padding=3,
                      external_flux=input.EXTERNAL_FLUX,
                      three_dim=input.THREE_DIM,
                      verbose=input.VERBOSE)

    model.run(input.EPOCHS)

    if input.SEEDS == 1:
        ANLS.fractal_dimension_analysis(LATTICE, 
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

    if input.SEEDS != 1:
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
    if input.THREE_DIM:
        logger.warning(f"""
        **************************************************************************************
        [MAIN LOOP] WARNING: 3D simulation is not implemented for Kobayashi and Stefan model!
        TERMINATING THE PROGRAM...
        **************************************************************************************
        """)
        return

    LATTICE = _init_phase_field_lattice(input)

    model = KobayashiGrowth(LATTICE,
                            epsilon0=input.EPSILON,
                            delta=input.DELTA,
                            n_folds=input.N_FOLDS,
                            mobility=input.MOBILITY,
                            supersaturation=input.SUPERSATURATION,
                            dt=input.TIME_STEP,
                            external_flux=None,
                            three_dim=False,
                            verbose=input.VERBOSE)

    model.run(input.EPOCHS)
    
    GUI.plot_phase_field_simulation(LATTICE,
                                    out_dir=input.OUTPUT_DIR,
                                    field_name="phi",
                                    color_field_name="phi",
                                    title=input.TITLE,
                                    three_dim=False)
    GUI.plot_phase_field_simulation(LATTICE,
                                    out_dir=input.OUTPUT_DIR,
                                    field_name="history",
                                    color_field_name="history",
                                    title=input.TITLE,
                                    three_dim=False)
    if input.SEEDS != 1:
        GUI.plot_kinetic_lattice(LATTICE, 
                         input.EPOCHS, 
                         title=input.TITLE+"_id", 
                         three_dim=False, 
                         out_dir=input.OUTPUT_DIR,
                         color_mode="id")
        GUI.plot_kinetic_lattice(LATTICE, 
                         input.EPOCHS,
                         title=input.TITLE+'_boundaries', 
                         three_dim=False, 
                         out_dir=input.OUTPUT_DIR, 
                         color_mode="boundaries")


def perform_STEFAN_simulation(input: custom_input):
    if input.THREE_DIM:
        logger.warning(f"""
        **************************************************************************************
        [MAIN LOOP] WARNING: 3D simulation is not implemented for Kobayashi and Stefan model!
        TERMINATING THE PROGRAM...
        **************************************************************************************
        """)
        return
    
    LATTICE = _init_phase_field_lattice(input)

    model = StefanGrowth(LATTICE,
                            epsilon0=input.EPSILON,
                            delta=input.DELTA,
                            n_folds=input.N_FOLDS,
                            mobility=input.MOBILITY,
                            diffusivity=input.DIFFUSIVITY,
                            latent_coeff=input.LATENT_COEF,
                            alpha=input.ALPHA,
                            gamma=input.GAMMA,
                            u_eq=input.U_EQ,
                            u_infty=input.U_INFTY,
                            enforce_dirichlet_u=True,
                            dt=input.TIME_STEP,
                            external_flux=None,
                            three_dim=False,
                            verbose=input.VERBOSE)

    model.run(input.EPOCHS)
    
    GUI.plot_phase_field_simulation(LATTICE,
                                    out_dir=input.OUTPUT_DIR,
                                    field_name = "phi",
                                    color_field_name="phi",
                                    title=input.TITLE,
                                    three_dim=False)
    GUI.plot_phase_field_simulation(LATTICE,
                                    out_dir=input.OUTPUT_DIR,
                                    field_name="u",
                                    color_field_name="u",
                                    title=input.TITLE,
                                    three_dim=False)
    GUI.plot_phase_field_simulation(LATTICE,
                                    out_dir=input.OUTPUT_DIR,
                                    field_name="history",
                                    color_field_name="history",
                                    title=input.TITLE,
                                    three_dim=False)
    if input.SEEDS != 1:
        GUI.plot_kinetic_lattice(LATTICE, 
                         input.EPOCHS, 
                         title=input.TITLE+"_id", 
                         three_dim=False, 
                         out_dir=input.OUTPUT_DIR,
                         color_mode="id")
        GUI.plot_kinetic_lattice(LATTICE, 
                         input.EPOCHS,
                         title=input.TITLE+'_boundaries', 
                         three_dim=False, 
                         out_dir=input.OUTPUT_DIR, 
                         color_mode="boundaries")

# TODO: rimuovere
def perform_active_surface_simulation(input: custom_input):
    LATTICE = KineticLattice(input.NX, input.NY, input.NZ, input.VERBOSE)
    for x in range(input.NX):
        for z in range(input.NZ):
            LATTICE.set_nucleation_seed(x, 0, z)

    # Default anisotropy: we aer simulating particles coming from +y direction
    flux = ParticleFlux(np.ndarray([0, 1, 0]), 
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
    