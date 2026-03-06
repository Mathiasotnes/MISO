import math
import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class SaliencyGrid(nn.Module):
    """
    Trainable 3D saliency grid as described in HollowNeRF.

    A Nx*Ny*Nz tensor of learnable scalar weights is trained alongside the
    NGP grid. Given a normalised coordinate x ∈ [0,1]^3. given a world-space 
    coordinate x, the saliency weight p ∈ (0,1) is:
        1. Map x to fractional grid coordinates.
        2. Trilinear interpolation of the 8 surrounding voxel corners.
        3. Sigmoid activation → p.

    The weighted feature v = p * f is then fed to the MLP decoder instead
    of the raw hash feature f. A feature with p → 0 is effectively pruned,
    reducing hash-collision interference for more important voxels.

    The grid is initialised to all-ones so that at the start of training the
    saliency weights are all sigmoid(1) ≈ 0.73, i.e. close to 1.
    """

    def __init__(
            self, 
            bound: torch.Tensor,
            res: float = 0.1,
            C: float = 0.1,
            rho: float = 1e-3,
            device: str = 'cuda:0'
        ):
        super().__init__()
        self.bound  = bound
        self.res    = res
        self.C      = C      # sparsity target: fraction of non-zero voxels
        self.rho    = rho    # learning rate for dual variable γ
        self.device = device
        
        self.Nx = math.ceil((bound[0, 1] - bound[0, 0]) / res)
        self.Ny = math.ceil((bound[1, 1] - bound[1, 0]) / res)
        self.Nz = math.ceil((bound[2, 1] - bound[2, 0]) / res)
        self.grid = nn.Parameter(torch.ones((self.Nx, self.Ny, self.Nz), device=device))

        # Corner offsets for trilinear interpolation.
        offsets = torch.tensor(
            [[i, j, k] for i in range(2) for j in range(2) for k in range(2)],
            dtype=torch.int32, device=device
        )  # (8, 3)
        self.register_buffer("corner_offsets", offsets)
        
        # Dual variable γ used for sparsity ADMM regularization
        self.register_buffer('gamma', torch.tensor(0.0, device=device))
        
        self.print_summary()
        
    def world_to_grid(self, x: torch.Tensor):
        """ Map world coords to fractional grid coordinates in [0, Nx/Ny/Nz]. """
        lo = self.bound[:, 0]
        hi = self.bound[:, 1]
        dims = torch.tensor([self.Nx, self.Ny, self.Nz], device=self.device, dtype=x.dtype)
        g_frac = (x - lo) / (hi - lo) * dims   # (N, 3)
        return g_frac
    
    def grid_to_index(self, g: torch.Tensor) -> torch.Tensor:
        """
        Convert integer grid coordinates to flat indices into the parameter tensor.

        Args:
            g: (N, 3) integer grid coordinates.
        Returns:
            (N,) flat indices.
        """
        gx = g[:, 0].clamp(0, self.Nx - 1)
        gy = g[:, 1].clamp(0, self.Ny - 1)
        gz = g[:, 2].clamp(0, self.Nz - 1)
        return gx + self.Nx * (gy + self.Ny * gz)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute saliency weights p for a batch of world-space coordinates.

        Args:
            x: (N, 3) world-space coordinates.
        Returns:
            p: (N, 1) saliency weights in (0, 1). Out-of-bound points receive p = 0
        """
        M = x.shape[0]
        grid_flat = self.grid.reshape(-1)           # (Nx*Ny*Nz,)

        g_frac = self.world_to_grid(x)              # (M, 3)
        x_floor = torch.floor(g_frac).long()        # (M, 3)
        w = g_frac - x_floor.float()                # (M, 3) fractional part

        # 8 corner integer coordinates and their flat indices
        corners = x_floor.unsqueeze(1) + self.corner_offsets.unsqueeze(0)  # (M, 8, 3)
        idx = self.grid_to_index(corners.reshape(M * 8, 3))                # (M*8,)
        corner_vals = grid_flat[idx].reshape(M, 8)                         # (M, 8)

        # Trilinear interpolation weights (same order as corner_offsets)
        wx0, wx1 = 1.0 - w[:, 0], w[:, 0]
        wy0, wy1 = 1.0 - w[:, 1], w[:, 1]
        wz0, wz1 = 1.0 - w[:, 2], w[:, 2]

        weights = torch.stack([
            wx0 * wy0 * wz0,    # (0,0,0)
            wx0 * wy0 * wz1,    # (0,0,1)
            wx0 * wy1 * wz0,    # (0,1,0)
            wx0 * wy1 * wz1,    # (0,1,1)
            wx1 * wy0 * wz0,    # (1,0,0)
            wx1 * wy0 * wz1,    # (1,0,1)
            wx1 * wy1 * wz0,    # (1,1,0)
            wx1 * wy1 * wz1,    # (1,1,1)
        ], dim=1)               # (M, 8)

        interpolated = (weights * corner_vals).sum(dim=1)   # (M,)
        return torch.sigmoid(interpolated).unsqueeze(-1)    # (M, 1)
    
    def sparsity(self) -> torch.Tensor:
        """ ||sigmoid(G)||_1 — the quantity we want to be < C. """
        return torch.sigmoid(self.grid).mean() - self.C
    
    def admm_loss(self) -> torch.Tensor:
        """
        Augmented Lagrangian term to add to the main loss.
        see HollowNeRF for details: https://arxiv.org/abs/2308.10122
        """
        s = self.sparsity()
        # (ρ/2) * [s]²₊  +  γ * s
        penalty = (self.rho / 2.0) * torch.clamp(s, min=0.0) ** 2
        lagrangian = self.gamma.detach() * s   # detach γ — it's updated separately
        return penalty + lagrangian
    
    @torch.no_grad()
    def update_gamma(self):
        s = self.sparsity()
        self.gamma.clamp_(min=0.0)
        self.gamma.add_(self.rho * s)
        self.gamma.clamp_(min=0.0)
    
    def print_summary(self):
        params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            f"\n{'='*40}\n"
            f" SaliencyGrid\n"
            f"   * Resolution         : {self.res} m\n"
            f"   * Grid dim           : ({self.Nx}, {self.Ny}, {self.Nz})\n"
            f"   * Trainable params   : {params:,}\n"
            f"{'='*40}"
        )
        