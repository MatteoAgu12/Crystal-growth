import argparse
import sys
import os

ALLOWED_NATIVE_SIMULATION_OPTIONS = ["EDEN", "DLA", "SURFACE", "POLI"]

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
    if not hasattr(parsed_input, 'epochs'):               parsed_input.epochs = 1000            # TODO: qui niente default: se non ci sono niente simulazione
    if not hasattr(parsed_input, 'size'):                 parsed_input.size = [100, 100, 100]   # TODO: qui niente default: se non ci sono niente simulazione
    if not hasattr(parsed_input, 'simulation'):           parsed_input.simulation = 'EDEN'      # TODO: qui niente default: se non ci sono niente simulazione
    if not hasattr(parsed_input, 'two_dim'):              parsed_input.two_dim = False
    if not hasattr(parsed_input, 'title'):                parsed_input.title = "Crystal lattice"
    if not hasattr(parsed_input, 'verbose'):              parsed_input.verbose = False
    if not hasattr(parsed_input, 'output'):               parsed_input.output = ""
    if not hasattr(parsed_input, 'external_flux'):        parsed_input.external_flux = None
    if not hasattr(parsed_input, 'flux_strength'):        parsed_input.flux_strength = 0.0
    if not hasattr(parsed_input, 'miller'):               parsed_input.miller = [0, 0, 0]
    if not hasattr(parsed_input, 'base_stick'):           parsed_input.base_stick = 0.01
    if not hasattr(parsed_input, 'anisotropy_coeff'):     parsed_input.anisotropy_coeff = 0.05
    if not hasattr(parsed_input, 'anisotropy_sharpness'): parsed_input.anisotropy_sharpness = 4.0
    if not hasattr(parsed_input, 'anisotropy_selection'): parsed_input.anisotropy_selection = 1.0
    if not hasattr(parsed_input, 'record'):               parsed_input.record = False

    # =========================================================================
    # VALIDATION CHECKS
    # =========================================================================
    # Epochs
    if not isinstance(parsed_input.epochs, int) or parsed_input.epochs < 0:
        raise ValueError(f"ERROR: 'epochs' must be a non-negative integer. Got: {parsed_input.epochs}")

    # Size
    if not isinstance(parsed_input.size, list) or len(parsed_input.size) != 3:
        raise ValueError(f"ERROR: 'size' must be a list of 3 integers (X Y Z). Got: {parsed_input.size}")
    for val in parsed_input.size:
        if not isinstance(val, int) or val <= 0:
            raise ValueError(f"ERROR: dimensions in 'size' must be positive integers. Got: {parsed_input.size}")

    # Simulation Type
    if parsed_input.simulation not in ALLOWED_NATIVE_SIMULATION_OPTIONS:
        raise ValueError(f"ERROR: 'simulation' must be one of {ALLOWED_NATIVE_SIMULATION_OPTIONS}. Got: {parsed_input.simulation}")

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
                    f"ERROR: '--external-flux' expects values in groups of 3 (AX AY AZ ...). "
                    f"You provided {len(vals)} values: {vals}")
            
            dirs = []
            for i in range(0, len(vals), 3):
                dirs.append([vals[i], vals[i+1], vals[i+2]])
            parsed_input.external_flux = dirs

    # Flux Strength
    if parsed_input.flux_strength < 0.0:
        raise ValueError(f"ERROR: 'flux-strength' cannot be negative. Got: {parsed_input.flux_strength}")

    # Miller Indices
    if not isinstance(parsed_input.miller, list) or len(parsed_input.miller) != 3:
        raise ValueError(f"ERROR: 'miller' must be a list of 3 integers. Got: {parsed_input.miller}")
    for val in parsed_input.miller:
        if not isinstance(val, int) or val < 0:
             raise ValueError(f"ERROR: Miller indices must be integers >= 0. Got: {parsed_input.miller}")

    # Anisotropy Parameters
    if parsed_input.anisotropy_coeff < 0.0:
        raise ValueError(f"ERROR: 'anisotropy-coeff' cannot be negative. Got: {parsed_input.anisotropy_coeff}")
    
    if parsed_input.anisotropy_sharpness < 0.0:
        raise ValueError(f"ERROR: 'anisotropy-sharpness' cannot be negative. Got: {parsed_input.anisotropy_sharpness}")
    
    if parsed_input.anisotropy_selection < 0.0:
        raise ValueError(f"ERROR: 'anisotropy-selection' cannot be negative. Got: {parsed_input.anisotropy_selection}")

    # Base Stick Probability
    if parsed_input.base_stick < 0.0 or parsed_input.base_stick > 1.0:
        raise ValueError(f"ERROR: 'base-stick' probability must be in range [0,1]. Got: {parsed_input.base_stick}")

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
        parsed_input = parse_inputs_from_terminal()
    
    # Checks the inputs
    check_parsed_inputs(parsed_input)
    
    return parsed_input
