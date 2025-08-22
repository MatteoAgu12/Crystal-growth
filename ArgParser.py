import argparse

def parse_inputs() -> argparse.Namespace:
    """
    Custom argument parser for the crystal structure symulation.
    
    Raises:
        ValueError: for each possible input check if its value is allowed.

    Returns:
        argparse.Namespace: parsed inputs.
    """
    ALLOWED_SIM_OPTIONS = ["EDEN", "DLA", "SURFACE"]
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
    
    # Checks the inputs
    parsed_input = parser.parse_args()
    if parsed_input.epochs < 0 or type(parsed_input.epochs) is not int:
        raise ValueError(f"ERROR: in function 'parse_inputs()', {parsed_input.epochs} is not a valid epoch value.")
    for input in parsed_input.size:
        if input <= 0 or type(input) is not int:
            raise ValueError(f"ERROR: in function 'parse_inputs()', {parsed_input.size} is not a valid size value.")    
    if parsed_input.simulation not in ALLOWED_SIM_OPTIONS:
        raise ValueError(f"ERROR: in function 'parse_inputs()', {parsed_input.simulation} is not a valid simulation option.")
    
    return parsed_input
