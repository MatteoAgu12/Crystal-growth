import argparse
import sys
import os

import logging
logger = logging.getLogger("growthsim")

ALLOWED_NATIVE_SIMULATION_OPTIONS = ["EDEN", "DLA", "KOBAYASHI", "STEFAN"]

def check_parsed_inputs(parsed_input: argparse.Namespace):
    """
    Checks if the numerical values and/or the number of inputs are correct for each input parameter.
    Ensures all attributes exist (setting defaults if missing) to prevent crashes when loading incomplete config files.
    Raises a ValueError if something is wrong.

    Args:
        parsed_input (argparse.Namespace): parser object from argparse.ArgumentParser.parse_args()
    """

    # =========================================================================
    # SAFETY DEFAULTS
    # =========================================================================
    failure_message = f"""
    |==========================================================================|
    | !!! FATAL ERROR IN INPUT !!!                                             |
    |                                                                          |
    | The following parameters are essentials and cannot be omitted:           |
    |    * epochs                                                              |      
    |    * size                                                                |   
    |    * simulation                                                          |
    |                                                                          |  
    |==========================================================================|
    """

    if (not hasattr(parsed_input, 'epochs') 
        or not hasattr(parsed_input, 'size') 
        or not hasattr(parsed_input, 'simulation')): 
            raise ValueError(f"{failure_message}")

    if not hasattr(parsed_input, 'output'):               parsed_input.output = None
    if not hasattr(parsed_input, 'two_dim'):              parsed_input.two_dim = False
    if not hasattr(parsed_input, "seeds"):                parsed_input.seeds = 1
    if not hasattr(parsed_input, 'title'):                parsed_input.title = "Crystal lattice"
    if not hasattr(parsed_input, 'verbose'):              parsed_input.verbose = False
    if not hasattr(parsed_input, 'external_flux'):        parsed_input.external_flux = None
    if not hasattr(parsed_input, 'flux_strength'):        parsed_input.flux_strength = 0.0

    if not hasattr(parsed_input, 'interface_thr'):        parsed_input.interface_thr = 0.5          
    if not hasattr(parsed_input, 'epsilon0'):             parsed_input.epsilon0 = 0.0           
    if not hasattr(parsed_input, 'delta'):                parsed_input.delta = 0.0          
    if not hasattr(parsed_input, 'n_folds'):              parsed_input.n_folds = 0.0            
    if not hasattr(parsed_input, 'alpha'):                parsed_input.alpha = 0.0          
    if not hasattr(parsed_input, 'u_equilibrium'):        parsed_input.u_equilibrium = 1.0        
    if not hasattr(parsed_input, 'diffusivity'):          parsed_input.diffusivity = 0.0 
    if not hasattr(parsed_input, 'mobility'):             parsed_input.mobility = 0.0
    if not hasattr(parsed_input, 'latent_coef'):          parsed_input.latent_coef = 0.0
    if not hasattr(parsed_input, 'gamma'):                parsed_input.gamma = 10.0
    if not hasattr(parsed_input, 'u_infinity'):           parsed_input.u_infinity = 0.0
    if not hasattr(parsed_input, 'supersaturation'):      parsed_input.supersaturation = 0.0            
    if not hasattr(parsed_input, 'dt'):                   parsed_input.dt = 1e-4            

    # =========================================================================
    # VALIDATION CHECKS
    # =========================================================================
    # Epochs
    if parsed_input.epochs < 0:
        raise ValueError(f"ERROR: 'epochs' must be a non-negative integer. Got: {parsed_input.epochs}")

    # Size
    if len(parsed_input.size) != 3:
        raise ValueError(f"ERROR: 'size' must be a list of 3 integers (X Y Z). Got: {parsed_input.size}")
    for val in parsed_input.size:
        if val <= 0:
            raise ValueError(f"ERROR: dimensions in 'size' must be positive integers. Got: {parsed_input.size}")

    # Simulation Type
    if parsed_input.simulation not in ALLOWED_NATIVE_SIMULATION_OPTIONS:
        raise ValueError(f"ERROR: 'simulation' must be one of {ALLOWED_NATIVE_SIMULATION_OPTIONS}. Got: {parsed_input.simulation}")

    # Initial nucleation seeds
    if parsed_input.seeds <= 0:
        raise ValueError(f"ERROR: 'seeds' must be an integer > 0. Got {parsed_input.seeds}")
    if parsed_input.seeds > 20:
        logger.warning("""
        ************************************************************************
         WARNING:
        \tThe GUI supports at most 20 different colors.
        \tWith %d some will be repeted.
        ************************************************************************
        """, parsed_input.seeds)

    # External Flux Logic
    if parsed_input.external_flux is not None:
        vals = parsed_input.external_flux
        if isinstance(vals, list) and not all(isinstance(x, list) for x in vals):
            if len(vals) % 3 != 0:
                raise ValueError(
                    f"ERROR: 'external-flux' expects values in groups of 3 (AX AY AZ ...). "
                    f"You provided {len(vals)} values: {vals}")
            
            dirs = []
            for i in range(0, len(vals), 3):
                dirs.append([vals[i], vals[i+1], vals[i+2]])
            parsed_input.external_flux = dirs

    # Flux Strength
    if parsed_input.flux_strength < 0.0:
        raise ValueError(f"ERROR: 'flux-strength' cannot be negative. Got: {parsed_input.flux_strength}")

    # Kobayashi parameters
    if (parsed_input.interface_thr      < 0.0 or
        parsed_input.epsilon0           < 0.0 or    
        parsed_input.delta              < 0.0 or        
        parsed_input.n_folds            < 0.0 or      
        parsed_input.alpha              < 0.0 or        
        parsed_input.u_equilibrium      < 0.0 or  
        parsed_input.u_equilibrium      > 1.0 or
        parsed_input.u_infinity         > 1.0 or       
        parsed_input.diffusivity        < 0.0 or  
        parsed_input.mobility           < 0.0 or  
        parsed_input.supersaturation    < 0.0 or  
        parsed_input.latent_coef        < 0.0 or
        parsed_input.gamma              < 0.0 or
        parsed_input.dt                 < 0):
            raise ValueError(f"ERROR: all the parameters in the PhaseField model (except latent_coef) must be >= 0.0\n \
                                      dt must be > 0.0\n \
                                      u_equilibrium and u_infinity must be in [0.0, 1.0].")

    # Delta VS N_folds
    max_delta = 1.0 / (parsed_input.n_folds**2 - 1)
    if (parsed_input.delta >= max_delta ) and parsed_input.simulation == "KOBAYASHI":
        logger.warning("""
        ************************************************************************************
         WARNING:
        \tDelta should be less than (1 / (n_fold**2 - 1)) to avoid negative stifness.
        \tThe value %f may cause sharp tips and other artifacts at the interface.
        ************************************************************************************
        """, parsed_input.delta)


    # Integration time
    if parsed_input.dt >= 5e-2:
        logger.warning("""
        ************************************************************************************
         WARNING:
        \tThe integration time should be kept small (~1e-4) to have high realism.
        \tThe value %f may cause numerical instability.
        ************************************************************************************
        """, parsed_input.dt)

def cast_file_input_to_value(value: str):
    """
    Convert a string value from a config file to an int, float, bool, list or str.

    Args:
        value (str): "value" of the input from the file.
    """
    value = value.strip()
    
    # bool
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    
    # list
    if " " in value or "," in value:
        sep = "," if "," in value else " "
        parts = [v for v in value.split(sep) if v]
        
        # int list
        try:
            return [int(v) for v in parts]
        except ValueError:
            pass
        
        # float list
        try:
            return [float(v) for v in parts]
        except ValueError:
            pass
        
        # default: str list
        return parts
    
    # int
    try:
        return int(value)
    except ValueError:
        pass
    
    # float 
    try:
        return float(value)
    except ValueError:
        pass
    
    # string
    return value

def parse_inputs_from_file_ini(filename: str) -> argparse.Namespace:
    """
    Custom argument parser for the crystal structure symulation.
    Does it by reading a .ini configuration file
    
    Args:
        filename (str): name of the configuration file.

    Returns:
        argparse.Namespace: parsed inputs.
    """
    params = {}

    with open(filename) as f:
        for raw_line in f:
            line = raw_line.strip()

            if not line or line.startswith("#") or line.startswith(";"):
                continue

            for comment_char in ("#", ";"):
                if comment_char in line:
                    line = line.split(comment_char, 1)[0].rstrip()

            if not line:
                continue

            if "=" not in line:
                raise ValueError(f"Invalid .ini line (missing '='): {raw_line.rstrip()}")

            key, value = line.split("=", 1)
            params[key.strip()] = cast_file_input_to_value(value.strip())

    return argparse.Namespace(**params)


def parse_inputs() -> argparse.Namespace:
    """
    Custom argument parser for the crystal structure symulation.
    It is a wrapper function, that decides which parser to call. Also checks the values of the inputs

    Returns:
        argparse.Namespace: parsed inputs.
    """
    from_file = True
    parsed_input = None
    
    if len(sys.argv) == 2 and sys.argv[1].endswith(".ini"):
        parsed_input = parse_inputs_from_file_ini(sys.argv[1])
    else:
        raise ValueError(f"""
    *************************************************************************
    * ERROR: No valid input file provided.                                  *
    * Please provide a .ini configuration file as a command line argument,  *
    * e.g.: python main.py config.ini                                       *
    *                                                                       *
    * Example of a valid config.ini file:                                   *
    * --------------------------------                                      *
    *   # Simulation settings                                               *
    *   simulation = KOBAYASHI                                              *
    *   size = 100 100 100                                                  *
    *   seeds = 5                                                           *
    *   epochs = 1000                                                       *
    *   output = ./output/                                                  *
    *   verbose = True                                                      *
    *                                                                       *
    *   # Kobayashi model parameters                                        *
    *   interface_thr = 0.5                                                 *
    *   epsilon0 = 1.0                                                      *
    *   delta = 0.2                                                         *
    *   n_folds = 4                                                         *
    *   alpha = 0.9                                                         *
    *   u_equilibrium = 0.5                                                 *
    *   u_infinity = 0.0                                                    *
    *   diffusivity = 1.0                                                   *
    *   mobility = 1.0                                                      *
    *   supersaturation = 0.2                                               *
    *   latent_coef = 1.0                                                   *
    *   gamma = 10.0                                                        *
    *   dt = 1e-4                                                           *
    *************************************************************************
        """)
    
    # Checks the inputs
    check_parsed_inputs(parsed_input)
    
    return parsed_input
