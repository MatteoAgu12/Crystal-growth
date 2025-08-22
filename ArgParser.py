import argparse

def parse_inputs() -> argparse.Namespace:
    """
    Custom argument parser for the crystal structure symulation.

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
    parser.add_argument("--2d", dest="three_dim", action="store_true",
                        help="Simulate a 2D crystal instead of a 3D one")
    
    return parser.parse_args()

print(type(parse_inputs()))