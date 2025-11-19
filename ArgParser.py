import argparse
import os

def parse_inputs() -> argparse.Namespace:
    """
    Custom argument parser for the crystal structure symulation.
    
    Raises:
        ValueError: for each possible input check if its value is allowed.

    Returns:
        argparse.Namespace: parsed inputs.
    """
    ALLOWED_SIM_OPTIONS = ["EDEN", "DLA", "SURFACE", "POLI"]
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
                        help=f"Type of built-in simulation to perform. Default 'EDEN'.\nAllowed options are {ALLOWED_SIM_OPTIONS}.")
    parser.add_argument("--verbose", "-v", dest="verbose", action="store_true",
                        help="If active prints extra info during the simulation.")
    parser.add_argument("--output", "-o", type=str, default="",
                        help="Directory where to save the plots produced.\nIf not specified, the analysis is not performed, only the simulation.")
    parser.add_argument("--anisotropy-directions", "-ad", type=float, nargs="+", metavar="AX AY AZ...", default=None,
                    help="List of anisotropy directions in group of 3: AX1 AY1 AZ1 AX2 AY2 AZ2 ... Default to None.")
    parser.add_argument("--anisotropy-strength", "-as", type=float, default=0.0,
                        help="Anisotropy strength (0.0 is isotropic). Default is 0.0")
    
    # Checks the inputs
    parsed_input = parser.parse_args()
    if parsed_input.epochs < 0 or type(parsed_input.epochs) is not int:
        raise ValueError(f"ERROR: in function 'parse_inputs()', {parsed_input.epochs} is not a valid epoch value.")
    for input in parsed_input.size:
        if input <= 0 or type(input) is not int:
            raise ValueError(f"ERROR: in function 'parse_inputs()', {parsed_input.size} is not a valid size value.")    
    if parsed_input.simulation not in ALLOWED_SIM_OPTIONS:
        raise ValueError(f"ERROR: in function 'parse_inputs()', {parsed_input.simulation} is not a valid simulation option.")
    if not parsed_input.output == "":
        if not os.path.isdir(parsed_input.output):
            raise ValueError(f"ERROR: in function 'parse_inputs()', directory {parsed_input.output} does not exist.")
    if parsed_input.anisotropy_directions is not None:
        vals = parsed_input.anisotropy_directions
        if len(vals) % 3 != 0:
            raise ValueError(
                f"ERROR: option '--anisotropy-directions' expects a number of values multiple of 3 "
                f"(AX AY AZ ...). You provided {len(vals)} values: {vals}")
        dirs = []
        for i in range(0, len(vals), 3):
            dirs.append([vals[i], vals[i+1], vals[i+2]])
        parsed_input.anisotropy_directions = dirs
    if parsed_input.anisotropy_strength < 0.0:
        raise ValueError("The anisotropy strength can't be negative.")
    
    return parsed_input
