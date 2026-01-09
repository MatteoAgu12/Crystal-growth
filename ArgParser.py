import argparse
import os

ALLOWED_NATIVE_SIMULATION_OPTIONS = ["EDEN", "DLA", "SURFACE", "POLI"]

def check_parsed_inputs(parsed_input: argparse.Namespace):
    """
    Checks if the numerical values and/or the number of inputs are correct for each input parameter.
    Raises a ValueError if something is wrong.

    Args:
        parser (argparse.Namespace): parser object from argparse.ArgumentParser.parse_args()
    """
    
    if parsed_input.epochs < 0 or type(parsed_input.epochs) is not int:
        raise ValueError(f"ERROR: in function 'parse_inputs()', {parsed_input.epochs} is not a valid epoch value.")
    for input in parsed_input.size:
        if input <= 0 or type(input) is not int:
            raise ValueError(f"ERROR: in function 'parse_inputs()', {parsed_input.size} is not a valid size value.")    
    if parsed_input.simulation not in ALLOWED_NATIVE_SIMULATION_OPTIONS:
        raise ValueError(f"ERROR: in function 'parse_inputs()', {parsed_input.simulation} is not a valid simulation option.")
    if not parsed_input.output == "":
        if not os.path.isdir(parsed_input.output):
            raise ValueError(f"ERROR: in function 'parse_inputs()', directory {parsed_input.output} does not exist.")
    if parsed_input.external_flux is not None:
        vals = parsed_input.external_flux
        if len(vals) % 3 != 0:
            raise ValueError(
                f"ERROR: option '--external-flux' expects a number of values multiple of 3 "
                f"(AX AY AZ ...). You provided {len(vals)} values: {vals}")
        dirs = []
        for i in range(0, len(vals), 3):
            dirs.append([vals[i], vals[i+1], vals[i+2]])
        parsed_input.external_flux = dirs
    if parsed_input.flux_strength < 0.0:
        raise ValueError("The external flux strength can't be negative.")
    for i in parsed_input.miller:
        if i < 0 or type(i) is not int:
            raise ValueError(f"ERROR: in function 'parse_inputs()', {parsed_input.miller} are not valid miller indices.")
    if parsed_input.anisotropy_coeff < 0.0:
        raise ValueError("The anisotropy coefficient can't be negative.")
    if parsed_input.anisotropy_sharpness < 0.0:
        raise ValueError("The anisotropy sharpness can't be negative.")

def parse_inputs() -> argparse.Namespace:
    """
    Custom argument parser for the crystal structure symulation.
    
    Raises:
        ValueError: for each possible input check if its value is allowed.

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
    parser.add_argument("--anisotropy-coeff", type=float, default=0.05,
                        help="Sticking coefficient that tells how easily a particle attaches to the surface due to anisotropy. Default is 0.05")
    parser.add_argument("--anisotropy-sharpness", type=float, default=4.0,
                        help="Tells how smooth are the faces due to anisotropy. Default is 4.0, cannot be less than 1.0")
    
    # Checks the inputs
    parsed_input = parser.parse_args()
    check_parsed_inputs(parsed_input)
    
    return parsed_input
