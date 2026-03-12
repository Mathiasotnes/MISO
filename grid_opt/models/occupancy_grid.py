import math
import torch
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class OccupancyGrid:
    """ A lightweight/simple occupancy grid. """
    def __init__(self, bound, res=0.05, device='cuda:0'):
        self.bound = bound
        self.res = res
        self.device = device
        self.Nx = math.ceil((self.bound[0,1] - self.bound[0,0]) / self.res)
        self.Ny = math.ceil((self.bound[1,1] - self.bound[1,0]) / self.res)
        self.Nz = math.ceil((self.bound[2,1] - self.bound[2,0]) / self.res)
        self.N = self.Nx * self.Ny * self.Nz
        self.grid = torch.zeros(self.N, dtype=torch.bool, device=device)
        self.print_summary()
    
    def world_to_grid(self, x: torch.Tensor) -> torch.Tensor:
        """ Convert world coordinates to grid coordinates.
        Args:
            x: (N, 3) tensor of world coordinates
        Returns:
            (N, 3) tensor of grid coordinates
        """
        # in-bounds mask in world space
        mask = (
            (x[:, 0] >= self.bound[0, 0]) & (x[:, 0] < self.bound[0, 1]) &
            (x[:, 1] >= self.bound[1, 0]) & (x[:, 1] < self.bound[1, 1]) &
            (x[:, 2] >= self.bound[2, 0]) & (x[:, 2] < self.bound[2, 1])
        )

        if not mask.any():
            return None, mask

        g = (x[mask] - self.bound[:, 0]) / self.res
        g = torch.floor(g).long()
        return g, mask
    
    def grid_to_index(self, g: torch.Tensor) -> torch.Tensor:
        """ Convert grid coordinates to grid index. """
        return g[:, 0] + self.Nx * (g[:, 1] + self.Ny * g[:, 2])
    
    @torch.no_grad()
    def update(self, x: torch.Tensor, sdf: torch.Tensor, tau: float = 0.1):
        """ Update the occupancy grid based on the input world coordinates and their corresponding SDF values.

        Args:
            x (torch.Tensor):           (N,3) tensor of world coordinates corresponding to the SDF values.
            sdf (torch.Tensor):         (N,) tensor of SDF values corresponding to the input world coordinates.
            tau (float, optional):      Threshold SDF value for marking cell as occupied. Defaults to 0.1.
        """
        sdf = sdf.view(-1)
        occ = sdf < tau
        
        g = self.world_to_grid(x)
        g, inb = self.world_to_grid(x)
        
        if g is None:
            return
        
        valid = occ[inb]
        if not valid.any():
            return

        idx = self.grid_to_index(g)
        self.grid[idx[valid]] = True

    def get_occupancy(self, x: torch.Tensor) -> torch.Tensor:
        """ Get the occupancy status of the grid cells corresponding to the input world coordinates.
        Args:
            x: (N, 3) tensor of world coordinates
        Returns:
            (N,) tensor of occupancy status (True for occupied, False for free)
        """
        g, inb = self.world_to_grid(x)
        out = torch.zeros(x.shape[0], device=x.device, dtype=torch.bool)
        if g is None:
            return out
        idx = self.grid_to_index(g)
        out[inb] = self.grid[idx]
        return out
    
    def print_summary(self):
        memory_kb = self.N / 1024
        logger.info(
            f"\n{'='*40}\n"
            f" OccupancyGrid\n"
            f"   * Resolution : {self.res} m\n"
            f"   * Dimensions : {self.Nx} x {self.Ny} x {self.Nz}\n"
            f"   * Cells      : {self.N:,}\n"
            f"   * Memory     ≈ {memory_kb:.1f} KB\n"
            f"{'='*40}"
        )
