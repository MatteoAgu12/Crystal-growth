import os
import numpy as np
import itertools
from typing import Union
import matplotlib.pyplot as plt

from classes.BaseLattice import BaseLattice
from GUI.gui_kinetic import get_visible_voxels_binary_mask, get_grain_boundaries_mask

import logging
logger = logging.getLogger("growthsim")

class KineticLattice(BaseLattice):
    """
    This class represents a specific implementation of the BaseLattice for kinetic crystal growth simulations.
    It defines the grid structure and methods for occupying cells, checking occupation status, and managing nucleation seeds.
    The lattice is represented as a 3D grid, where each cell can be occupied or unoccupied. 
    It also keeps track of the history of occupation and group IDs for nucleation processes.
    """
    def __init__(self, nx: int, ny: int, nz: int, verbose: bool = False):
        """
        Args:
            nx (int): number of cells in the lattice along the x direction
            ny (int): number of cells in the lattice along the y direction
            nz (int): number of cells in the lattice along the z direction
            verbose (bool, optional): if True, the lattice will print debug information during initialization and occupation of cells. Defaults to False.
        """
        super().__init__(nx, ny, nz, verbose)
        if nx < 0 or ny < 0 or nz < 0:
            raise ValueError('ERROR: the size of the lattice must be an integer bigger or equal to zero!')

        self.grid    = np.zeros(self.shape, dtype=np.uint8)
        logger.debug("%s", self)

    def __str__(self):
        return super().__str__()
    
    def occupy(self, x: int, y: int, z: int, epoch: int, id: int) -> None:
        """
        Set the status of the cell at coordinates (x,y,z) to occupied.
        It also keeps track of the epoch at which the new cell has been occupied.
        
        Args:
            x (int): x coordinate of the point to occupy
            y (int): y coordinate of the point to occupy
            z (int): z coordinate of the point to occupy
            epoch (int): current epoch in the simulation
            id (int): group id to assign to the occupied cell
            
        Raises:
            ValueError: if the epoch number is negative the function raises error.
        """
        if epoch < 0:
            raise ValueError(f"ERROR: in function 'KineticLattice::occupy()' the epoch number must be a an integer bigger or equal to zero.")
        
        if not self.is_point_inside(x, y, z):
            return
        
        if (x, y, z) in self.occupied:
            return
        
        self.grid[x, y, z] = 1
        self.history[x, y, z] = int(epoch)
        self.occupied.add((int(x), int(y), int(z)))
        self.group_id[x, y, z] = id
            
    def is_occupied(self, x: int, y: int, z: int) -> bool:
        """
        Function that check if a point at coordinates (x,y,z) i occupied or not.

        Args:
            x (int): x coord of the pixel
            y (int): y coord of the pixel
            z (int): z coord of the pixel
            
        Returns:
            (bool): if that pixel is occupied or not
        """
        return (int(x), int(y), int(z)) in self.occupied

    def get_group_id(self, x: int, y: int, z: int) -> int:
        """
        Function that returns the group id of cell at coordinates (x,y,z).

        Args:
            x (int): x coord of the pixel
            y (int): y coord of the pixel
            z (int): z coord of the pixel

        Returns:
            int: group id of the selected cell.
        """
        return self.group_id[x, y, z]

    def get_group_counter(self) -> int:
        """
        Function that return the total number of crystal groups in the lattice.

        Returns:
            int: total number of crystal groups in the lattice
        """
        return self.group_counter

    def set_nucleation_seed(self, x: int, y: int, z: int, maintain_last_id: bool = False) -> None:
        """
        Function that initialize one nucleation seed.
        The function does nothing if the selected point is outside the lattice or if it's already a nucleation seed.

        Args:
            x (int): x coordinate of the nucleation seed
            y (int): y coordinate of the nucleation seed
            z (int): z coordinate of the nucleation seed
            maintain_last_id (bool, optional): if True, it gives that nucleation seed the same group id than the previous one.
                                               If self.group_counter is 0, this is bypassed to True
        """
        if self.is_point_inside(x, y, z) and (x, y, z) not in self.initial_seeds:
            if self.group_counter == 0 or not maintain_last_id:
                self.group_counter += 1
            
            self.occupy(x, y, z, epoch=0, id=self.group_counter)
            self.initial_seeds.append((x, y, z))
    
    def get_nucleation_seeds(self) -> np.ndarray:
        """
        This function returns the initial nucleation seeds.

        Returns:
            np.ndarray: array containing all the initial nucleation seeds.
        """
        return np.array(self.initial_seeds)
    
    def get_active_border(self):
        """
        Function that compute the active border of teh crystal.
        The active border is the set of all empty cells having at least one occupied cell as neighbor.

        Returns:
            (np.ndarray): array containing the points that form the active border
        """
        active_set = set()
                
        for (x, y, z) in self.occupied:
            for (nx, ny, nz) in self.get_neighbors(x, y, z):
                if (int(nx), int(ny), int(nz)) not in self.occupied:
                    active_set.add((int(nx), int(ny), int(nz)))
                    
        if not active_set: return np.array([]).reshape((0, 3))
        return np.array(list(active_set), dtype=int)
 
    def get_crystal_bounding_box(self, padding: int = 0) -> Union[list, None]:
        """
        Function to compute the bounding box of the occupied region (smallest parallelogram that contains it).

        Args:
            padding (int, optional): optional enlargement (in each direction) of the box. Defaults to 0.

        Returns:
            (Union[list, None]): tuple containing the information (coord_min, coord_max) for each coordinate.
        """
        if not self.occupied:
            return None
        
        occupied_coords = np.array(list(self.occupied), dtype=int)
        mins = occupied_coords.min(axis=0) - padding
        maxs = occupied_coords.max(axis=0) + 1 + padding
        
        mins = np.clip(mins, 0, np.array(self.shape) - 1)
        maxs = np.clip(maxs, 0, np.array(self.shape) - 1)
        
        return list(zip(mins, maxs))

    def _save_frame_kinetic_2D(self, epoch: int, frame_dir: str, frame_list: list) -> str:
        """
        Saves a current frame of the 2D simulation.

        Args:
            epoch (int): current epoch number
            frame_dir (str): output directory
            frame_list (list): list of past frame to be updated with the current one

        Returns:
            str: path of the frame saved
        """
        z = 0 if self.shape[2] == 1 else self.shape[2] // 2

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        id_data = self.group_id[:, :, z].astype(float)
        id_masked = np.ma.masked_where(id_data == 0, id_data)
        axs[0].imshow(id_masked.T, origin='lower', cmap='tab20b', interpolation='nearest')
        axs[0].set_title(r"Group ID")

        hist = self.history[:,:,z].astype(float)
        hist_masked = np.ma.masked_where(hist < 0, hist)
        axs[1].imshow(hist_masked.T, origin='lower', cmap='turbo', interpolation='nearest')
        axs[1].set_title(r"Occupation history")

        boundary_mask_3d = get_grain_boundaries_mask(self)
        boundary_mask_2d = boundary_mask_3d[:, :, z]
        occ_2d = self.grid[:, :, z].astype(bool)

        nx, ny = occ_2d.shape
        rgba_img = np.ones((ny, nx, 4)) 
        rgba_img[occ_2d.T] = [0.9, 0.9, 0.9, 1.0]
        rgba_img[boundary_mask_2d.T] = [0.0, 0.0, 0.0, 1.0]

        axs[2].imshow(rgba_img, origin='lower', interpolation='nearest')
        axs[2].set_title("Grain boundaries")

        filepath = os.path.join(frame_dir, f"frame_{epoch:05d}.png")
        plt.savefig(filepath, bbox_inches='tight')
        plt.close(fig)

        frame_list.append(filepath)

    def _save_frame_kinetic_3D(self, epoch: int, frame_dir: str, frame_list: list) -> str:
        """
        Saves a current frame of the 3D simulation.

        Args:
            epoch (int): current epoch number
            frame_dir (str): output directory
            frame_list (list): list of past frame to be updated with the current one

        Returns:
            str: path of the frame saved
        """
        fig = plt.figure(figsize=(18, 6))

        visible_mask = get_visible_voxels_binary_mask(self)
        x, y, z = np.where(visible_mask)
        nx, ny, nz = self.shape

        if len(x) > 0:
            id_vals = self.group_id[x, y, z]
            hist_vals = self.history[x, y, z]
            boundaries_mask_global = get_grain_boundaries_mask(self)
            is_boundary = boundaries_mask_global[x, y, z]
            boundary_colors = np.where(is_boundary, 'black', 'lightgray')

        markersize = 15
        ax1 = fig.add_subplot(1, 3, 1, projection='3d')
        if len(x) > 0:
            ax1.scatter(x, y, z, c=id_vals, cmap='tab20', marker='s', s=markersize, alpha=1.0, edgecolor='none')
        ax1.set_title("Group ID")

        ax2 = fig.add_subplot(1, 3, 2, projection='3d')
        if len(x) > 0:
            ax2.scatter(x, y, z, c=hist_vals, cmap='turbo', marker='s', s=markersize, alpha=1.0, edgecolors='none')
        ax2.set_title("Occupation History")

        ax3 = fig.add_subplot(1, 3, 3, projection='3d')
        if len(x) > 0:
            ax3.scatter(x, y, z, c=boundary_colors, marker='s', s=markersize, alpha=1.0, edgecolors='none')
        ax3.set_title("Grain Boundaries")

        for ax in [ax1, ax2, ax3]:
            ax.set_xlim(0, nx)
            ax.set_ylim(0, ny)
            ax.set_zlim(0, nz)
            ax.view_init(elev=30, azim=45)

            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])

        filepath = os.path.join(frame_dir, f"frame_{epoch:05d}.png")
        plt.savefig(filepath, bbox_inches='tight', dpi=100) 
        plt.close(fig)

        frame_list.append(filepath)
        return filepath

    def save_frame(self, epoch: int, three_dim: bool, frame_dir: str, frame_list: list) -> str:
        """
        Wrapper function that calls the utility to save the current frame, depending on the number of dimentions.

        Args:
            epoch (int): current epoch number
            three_dim (bool): if True is 3D, else 2D
            frame_dir (str): output directory path
            frame_list (list): list containing the path of all the frames, to be updated

        Returns:
            str: the path of the produce frame
        """
        if three_dim:
            return self._save_frame_kinetic_3D(epoch, frame_dir, frame_list)
        else:
            return self._save_frame_kinetic_2D(epoch, frame_dir, frame_list)