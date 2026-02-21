from utils.ArgParser import parse_inputs
from utils.logger import setup_logging
from utils.paths import ensure_output_dir
from classes.ParticleFlux import ParticleFlux
import simulations as SIM 

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
    out_dir       = ensure_output_dir(parsed_inputs.output, SIMULATION)
    frame_freq    = parsed_inputs.frame_freq
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
    gamma           = parsed_inputs.gamma
    diffusivity     = parsed_inputs.diffusivity
    mobility        = parsed_inputs.mobility
    supersaturation = parsed_inputs.supersaturation
    time_step      = parsed_inputs.dt
    
    # ================================================================
    # Creating the input objects
    # ================================================================
    simulation_input = SIM.custom_input(SIMULATION=SIMULATION,
                                        NX=nx, NY=ny, NZ=nz,
                                        SEEDS=seeds,
                                        EPOCHS=epochs, 
                                        THREE_DIM=is_3D, 
                                        FRAME_FREQ=frame_freq,
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
                                        GAMMA=gamma,
                                        MOBILITY=mobility,
                                        DIFFUSIVITY=diffusivity,
                                        SUPERSATURATION=supersaturation,
                                        TIME_STEP=time_step)
    
    logger = setup_logging(verbose=simulation_input.VERBOSE, log_file=f"{simulation_input.OUTPUT_DIR}/run.log")
    logger.info("Starting simulation: %s", simulation_input.SIMULATION)
    logger.debug("Config:\n%s", simulation_input)

    # ================================================================
    # Run the desired simulation
    # ================================================================
    if SIMULATION == 'EDEN':
        SIM.perform_EDEN_simulation(simulation_input)
        
    elif SIMULATION == 'DLA':
        SIM.perform_DLA_simulation(simulation_input)

    elif SIMULATION == 'KOBAYASHI':
        SIM.perform_KOBAYASHI_simulation(simulation_input)

    elif SIMULATION == 'STEFAN':
        SIM.perform_STEFAN_simulation(simulation_input)

    else:
        logger.error("""
        *****************************************************************************
        [MAIN LOOP] ERROR: simulation mode %s is not a valid option!
        TERMINATING THE PROGRAM...
        *****************************************************************************
        """, SIMULATION)