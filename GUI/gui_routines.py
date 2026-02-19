import os
import matplotlib.pyplot as plt
from classes.BaseLattice import BaseLattice
import logging

logger = logging.getLogger("growthsim")

def _mid_plane_z(lattice: BaseLattice) -> int:
    """
    Return the mid-plane index along z.

    Args:
        lattice (BaseLattice): general lattice object

    Return:
        (int): z coordinate of the mid plane
    """
    return 0 if lattice.shape[2] == 1 else lattice.shape[2] // 2

def set_axes_labels(ax: plt.Axes, is_3d: bool = False) -> None:
    """
    Set the standard spatial labels for the plot axes.

    Args:
        ax (plt.Axes): matplotlib axes object
        is_3d (bool): flag to determine if the z-axis label should be set
    """
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    if is_3d:
        ax.set_zlabel("Z")

def finalize_plot(out_dir: Union[str, None], title: str, suffix: str, log_message: str) -> None:
    """
    Handle the repetitive logic of adjusting layout, saving the figure, logging, and displaying it.

    Args:
        out_dir (str or None): output directory to save the plot. If None, the plot is not saved
        title (str): title of the plot, used to generate the filename
        suffix (str): string appended to the end of the filename (e.g., color mode)
        log_message (str): message to print in the logger upon successful save
    """
    plt.tight_layout()
    if out_dir is not None:
        file_suffix = f"_{suffix}" if suffix else ""
        filename = os.path.join(out_dir, f"{title.replace(' ', '_')}{file_suffix}.png")
        plt.savefig(filename, bbox_inches='tight')
        logger.info(log_message, filename)
    plt.show()