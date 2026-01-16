import argparse
import sys
import os

ALLOWED_NATIVE_SIMULATION_OPTIONS = ["EDEN", "DLA", "KOBAYASHI", "SURFACE"]

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
    |    * output                                                              |
    |                                                                          |  
    |==========================================================================|
    """

    if (not hasattr(parsed_input, 'epochs') 
        or not hasattr(parsed_input, 'size') 
        or not hasattr(parsed_input, 'simulation') 
        or not hasattr(parsed_input, 'output')):    
            raise ValueError(f"{failure_message}")

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
    if not hasattr(parsed_input, 'tau'):                  parsed_input.tau = 1.0            
    if not hasattr(parsed_input, 'diffusivity'):          parsed_input.diffusivity = 0.0            
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
        print(f"""
        ************************************************************************
         ATTENTION:
        \tThe GUI supports at most 20 different colors.
        \tWith {parsed_input.seeds} some will be repeted.
        ************************************************************************
        """)

    # Output Directory
    if parsed_input.output != "":
        if not os.path.isdir(parsed_input.output):
            raise ValueError(f"ERROR: The output directory '{parsed_input.output}' does not exist.")

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
    if (parsed_input.interface_thr  < 0.0 or
        parsed_input.epsilon0       < 0.0 or    
        parsed_input.delta          < 0.0 or        
        parsed_input.n_folds        < 0.0 or      
        parsed_input.alpha          < 0.0 or        
        parsed_input.u_equilibrium  < 0.0 or
        parsed_input.tau           <= 0.0 or          
        parsed_input.diffusivity    < 0.0 or  
        parsed_input.dt             < 0):
            raise ValueError(f"ERROR: all the parameters in the Kobayashi growth must be >= 0.0\n \
                                      Only excpetion for dt, which must be > 0.0")

    # Integration time
    if parsed_input.dt >= 1e-2:
        print(f"""
        ************************************************************************************
         ATTENTION:
        \tThe integration time should be kept small (~1e-4) to have high realism.
        \tThe value {parsed_input.dt} may cause numerical instability.
        ************************************************************************************
        """)

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

# TODO: rimuovere questa funzione
def parse_inputs_from_terminal() -> argparse.Namespace:
    """
    Custom argument parser for the crystal structure symulation.
    Does it with argparser library (inputs from terminal)

    Returns:
        argparse.Namespace: parsed inputs.
    """
    parser = argparse.ArgumentParser(
        description="Crystal growth simulation"
    )
    
    # Arguments
    parser.add_argument("--epochs", "-e", type=int, default=1000,
                        help="Number of epochs in the simulation, default 1000")
    parser.add_argument("--size", "-s", type=int, nargs=3, metavar=("NX", "NY", "NZ"), default=[100, 100, 100],
                        help="Grid dimention, in the form: X Y Z. Default 100 100 100")
    parser.add_argument("--seeds", type=int, default=1,
                        help="Number of initial nucleation seeds That are randomly scattered. By default 1 at the center.")
    parser.add_argument("--2d", "--2D", dest="two_dim", action="store_true",
                        help="Simulate a 2D crystal instead of a 3D one")
    parser.add_argument("--title", "-t", type=str, default="Crystal lattice",
                        help="Simulation title, default 'Crystal lattice'.")
    parser.add_argument("--simulation", "--sim", type=str, default='EDEN',
                        help=f"Type of built-in simulation to perform. Default 'EDEN'.\nAllowed options are {ALLOWED_NATIVE_SIMULATION_OPTIONS}.")
    parser.add_argument("--verbose", "-v", dest="verbose", action="store_true",
                        help="If active prints extra info during the simulation.")
    parser.add_argument("--output", "-o", type=str, default="",
                        help="Directory where to save the plots produced.\nIf not specified, the analysis is not performed, only the simulation.")
    parser.add_argument("--external-flux", "--ef", type=float, nargs="+", metavar="AX AY AZ...", default=None,
                    help="List of directions for the external diffusion flux, in group of 3: AX1 AY1 AZ1 AX2 AY2 AZ2 ... Default to None.")
    parser.add_argument("--flux-strength", "--fs", type=float, default=0.0,
                        help="External diffusion flux strength (0.0 is isotropic). Default is 0.0")
    parser.add_argument("--miller", "-m", type=int, nargs=3, metavar=("h", "k", "l"), default=[0, 0, 0],
                        help="Miller indices that define the structural anisotropy directions. Default [0,0,0] (no anisotropy)")
    parser.add_argument("--base-stick", type=float, default=0.01,
                        help="Residual isotropic sticking probability, muste be in [0,1]. For strong anisotropy select values close to 0.")
    parser.add_argument("--anisotropy-coeff", type=float, default=0.05,
                        help="Sticking coefficient that tells how easily a particle attaches to the surface due to anisotropy. Default is 0.05")
    parser.add_argument("--anisotropy-sharpness", type=float, default=4.0,
                        help="Tells how smooth are the faces due to anisotropy. Default is 4.0, cannot be less than 1.0")
    parser.add_argument("--anisotropy-selection", type=float, default=1.0,
                        help="Selection parameter, tells how efficient the anisotropy is. Goes as exp(selection * ...)")
    parser.add_argument("--record", "-R", dest="record", action="store_true",
                        help="If active records some diagnostic information during the simulation.")
    
    parsed_input = parser.parse_args()
    return parsed_input
    
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
        for line in f:
            line = line.strip()
            
            if not line or line.startswith("#"):
                continue
            
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
        raise ValueError(f"FATAL ERROR: input {[sys.argv[1:]]} was not a correct file .ini")
    
    # Checks the inputs
    check_parsed_inputs(parsed_input)
    
    return parsed_input
