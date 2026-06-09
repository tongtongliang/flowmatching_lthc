"""Local Token Hyper-Connection JiT.

The persistent residual stream is a high-resolution NHWC token grid. Each block
reads a low-resolution temporary workspace from that stream, runs a normal
JiT-style attention/FFN branch on the workspace, then writes the branch update
back to local high-resolution cells. The workspace state itself is not
persisted across blocks.
"""

from __future__ import annotations

import math
import re
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from .jit_shared_adaln import RMSNorm, VisionRotaryEmbeddingFast, get_2d_sincos_pos_embed
from .local_thc_triton_kernels import local_read as triton_local_read
from .local_thc_triton_kernels import local_read_add as triton_local_read_add
from .local_thc_triton_kernels import local_read_add_no_weight_grad as triton_local_read_add_no_weight_grad
from .local_thc_triton_kernels import local_read_add_traceable as triton_local_read_add_traceable
from .local_thc_triton_kernels import local_write as triton_local_write
from .local_thc_triton_kernels import local_write_no_weight_grad as triton_local_write_no_weight_grad
from .local_thc_triton_kernels import local_write_traceable as triton_local_write_traceable
from .local_thc_triton_kernels import all_base_b4_12_traceable as triton_all_base_b4_12_traceable
from .local_thc_triton_kernels import triangular_accumulate_traceable as triton_triangular_accumulate_traceable
from .local_thc_triton_kernels import triangular_accumulate_b4_12_ptr_traceable as triton_triangular_accumulate_b4_12_ptr_traceable
from .local_thc_triton_kernels import triangular_accumulate_b4_12_ptr_active_traceable_ops as triton_triangular_active_ops
from .local_thc_triton_kernels import final_accumulate_b4_traceable as triton_final_accumulate_b4_traceable
from .local_thc_triton_kernels import final_accumulate_b4_12_traceable as triton_final_accumulate_b4_12_traceable


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def migrate_lthc_state_dict_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Map old read/write checkpoint keys to the README-consistent names.

    Older research checkpoints used ``write`` for residual -> workspace and
    ``read`` for workspace -> residual. The public code uses the opposite,
    semantically correct convention:

    - read: residual -> workspace, alpha/read_logits
    - write: workspace -> residual, beta/write_weight
    """
    migrated = {}
    for key, value in state_dict.items():
        new_key = key
        new_key = new_key.replace("shared_write.", "shared_read.")
        new_key = re.sub(r"blocks\.(\d+)\.read\.", r"blocks.\1.write.", new_key)
        new_key = new_key.replace(".hyper.write_logits", ".hyper.read_logits")
        new_key = new_key.replace(".hyper.read_weight", ".hyper.write_weight")
        new_key = new_key.replace(".write_logits", ".read_logits")
        new_key = new_key.replace(".read_weight", ".write_weight")
        migrated[new_key] = value
    return migrated


class BottleneckPatchEmbedNHWC(nn.Module):
    """Image-to-high-resolution patch tokens in NHWC layout."""

    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 4,
        in_chans: int = 3,
        bottleneck_dim: int = 128,
        embed_dim: int = 768,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.grid_size = (img_size // patch_size, img_size // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj1 = nn.Conv2d(in_chans, bottleneck_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.proj2 = nn.Conv2d(bottleneck_dim, embed_dim, kernel_size=1, stride=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        if (h, w) != self.img_size:
            raise ValueError(f"expected image size {self.img_size}, got {(h, w)}")
        x = self.proj2(self.proj1(x))
        return x.permute(0, 2, 3, 1).contiguous()


class DirectPatchEmbedNHWC(nn.Module):
    """Direct image-to-token patch embedding in NHWC layout.

    For patch_size=4 this maps each 4x4x3 patch directly to hidden_size. It
    removes the 3->bottleneck->hidden two-stage projection.
    """

    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 4,
        in_chans: int = 3,
        bottleneck_dim: int = 128,
        embed_dim: int = 768,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.grid_size = (img_size // patch_size, img_size // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        if (h, w) != self.img_size:
            raise ValueError(f"expected image size {self.img_size}, got {(h, w)}")
        return self.proj(x).permute(0, 2, 3, 1).contiguous()


class BottleneckPatchEmbedNCHW(nn.Module):
    """Image-to-high-resolution patch tokens in native Conv2d layout."""

    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 4,
        in_chans: int = 3,
        bottleneck_dim: int = 128,
        embed_dim: int = 768,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.grid_size = (img_size // patch_size, img_size // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj1 = nn.Conv2d(in_chans, bottleneck_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.proj2 = nn.Conv2d(bottleneck_dim, embed_dim, kernel_size=1, stride=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        if (h, w) != self.img_size:
            raise ValueError(f"expected image size {self.img_size}, got {(h, w)}")
        return self.proj2(self.proj1(x))


class DirectPatchEmbedNCHW(nn.Module):
    """Direct image-to-token patch embedding in native Conv2d layout."""

    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 4,
        in_chans: int = 3,
        bottleneck_dim: int = 128,
        embed_dim: int = 768,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.grid_size = (img_size // patch_size, img_size // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        if (h, w) != self.img_size:
            raise ValueError(f"expected image size {self.img_size}, got {(h, w)}")
        return self.proj(x)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.timestep_embedding(t, self.frequency_embedding_size))


class LabelEmbedder(nn.Module):
    def __init__(self, num_classes: int, hidden_size: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.embedding_table = nn.Embedding(num_classes + 1, hidden_size)

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        return self.embedding_table(labels)


class WorkspaceAttention(nn.Module):
    """Self-attention on the temporary low-resolution workspace only."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 12,
        qkv_bias: bool = True,
        qk_norm: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        attn_backend: str = "flash",
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_backend = attn_backend

    def _sdpa_context(self):
        if self.attn_backend == "flash":
            return sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH])
        if self.attn_backend == "efficient":
            return sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH])
        if self.attn_backend == "math":
            return sdpa_kernel([SDPBackend.MATH])
        return nullcontext()

    def forward(self, x: torch.Tensor, rope: VisionRotaryEmbeddingFast) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = rope(self.q_norm(q))
        k = rope(self.k_norm(k))
        dropout_p = self.attn_drop.p if self.training else 0.0
        with self._sdpa_context():
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=False)
        x = x.transpose(1, 2).reshape(b, n, c)
        return self.proj_drop(self.proj(x))


class SwiGLUFFN(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, drop: float = 0.0, bias: bool = True) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim * 2 / 3)
        self.w12 = nn.Linear(dim, 2 * hidden_dim, bias=bias)
        self.w3 = nn.Linear(hidden_dim, dim, bias=bias)
        self.ffn_dropout = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.w12(x).chunk(2, dim=-1)
        return self.w3(self.ffn_dropout(F.silu(x1) * x2))


class WorkspaceJiTBranch(nn.Module):
    """Normal per-block AdaLN branch on temporary workspace tokens."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        attn_backend: str = "flash",
    ) -> None:
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = WorkspaceAttention(hidden_size, num_heads, attn_drop=attn_drop, proj_drop=proj_drop, attn_backend=attn_backend)
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        self.mlp = SwiGLUFFN(hidden_size, int(hidden_size * mlp_ratio), drop=proj_drop)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))

    def forward(self, z: torch.Tensor, c: torch.Tensor, rope: VisionRotaryEmbeddingFast) -> torch.Tensor:
        return self.forward_with_modulation(z, self.adaLN_modulation(c), rope)

    def forward_with_modulation(self, z: torch.Tensor, t6: torch.Tensor, rope: VisionRotaryEmbeddingFast) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = t6.chunk(6, dim=-1)
        dz_attn = gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(z), shift_msa, scale_msa), rope)
        z_tmp = z + dz_attn
        dz_mlp = gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(z_tmp), shift_mlp, scale_mlp))
        return dz_attn + dz_mlp


class WorkspaceSharedAdaLNBranch(nn.Module):
    """Shared-AdaLN branch; modulation is produced once by the top-level model."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        attn_backend: str = "flash",
    ) -> None:
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = WorkspaceAttention(hidden_size, num_heads, attn_drop=attn_drop, proj_drop=proj_drop, attn_backend=attn_backend)
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        self.mlp = SwiGLUFFN(hidden_size, int(hidden_size * mlp_ratio), drop=proj_drop)

    def forward(self, z: torch.Tensor, t6: torch.Tensor, rope: VisionRotaryEmbeddingFast) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = t6.chunk(6, dim=-1)
        dz_attn = gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(z), shift_msa, scale_msa), rope)
        z_tmp = z + dz_attn
        dz_mlp = gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(z_tmp), shift_mlp, scale_mlp))
        return dz_attn + dz_mlp


class ChannelwiseLocalTokenHyperConnection(nn.Module):
    """Local-cell maps between the persistent residual stream and workspace.

    Naming convention in new code:
      - read_from_residual:  z = R_l x_l
      - write_to_residual:  dx = P_l dz_l

    Historical checkpoints used the reversed names ``write_logits`` and
    ``read_weight``. New checkpoints use ``read_logits`` for alpha and
    ``write_weight`` for beta; the loader migrates old keys automatically.
    """

    def __init__(
        self,
        high_grid: int,
        workspace_grid: int,
        hidden_size: int,
        pool_groups: int,
        use_softmax_write: bool = True,
        init_read: float = 1.0,
    ) -> None:
        super().__init__()
        if high_grid % workspace_grid != 0:
            raise ValueError(f"high_grid={high_grid} must be divisible by workspace_grid={workspace_grid}")
        if hidden_size % pool_groups != 0:
            raise ValueError(f"hidden_size={hidden_size} must be divisible by pool_groups={pool_groups}")
        self.high_grid = high_grid
        self.workspace_grid = workspace_grid
        self.hidden_size = hidden_size
        self.pool_groups = pool_groups
        self.group_dim = hidden_size // pool_groups
        self.cell_size = high_grid // workspace_grid
        self.cell_tokens = self.cell_size * self.cell_size
        self.use_softmax_write = use_softmax_write
        self.read_logits = nn.Parameter(torch.zeros(pool_groups, self.cell_tokens))
        self.write_weight = nn.Parameter(torch.full((pool_groups, self.cell_tokens), float(init_read)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.constant_(self.read_logits, 0.0)
        nn.init.constant_(self.write_weight, 1.0)

    def _residual_read_weight(self, dtype: torch.dtype) -> torch.Tensor:
        if self.use_softmax_write:
            w = F.softmax(self.read_logits.float(), dim=-1)
        else:
            w = self.read_logits
        return w.to(dtype=dtype).view(self.pool_groups, self.cell_size, self.cell_size)

    def _residual_write_weight(self, dtype: torch.dtype) -> torch.Tensor:
        return self.write_weight.to(dtype=dtype).view(self.pool_groups, self.cell_size, self.cell_size)

    def read_from_residual(self, x: torch.Tensor) -> torch.Tensor:
        b, h, w, c = x.shape
        hh = self.high_grid
        gw = self.workspace_grid
        s = self.cell_size
        gp = self.pool_groups
        d = self.group_dim
        if h != hh or w != hh or c != self.hidden_size:
            raise ValueError(f"expected [B,{hh},{hh},{self.hidden_size}], got {tuple(x.shape)}")
        x = x.view(b, gw, s, gw, s, gp, d)
        alpha = self._residual_read_weight(x.dtype).permute(1, 2, 0)
        alpha = alpha[None, None, :, None, :, :, None]
        z = (x * alpha).sum(dim=(2, 4))
        return z.reshape(b, gw * gw, c)

    def write_to_residual(self, dz: torch.Tensor) -> torch.Tensor:
        b, n, c = dz.shape
        hh = self.high_grid
        gw = self.workspace_grid
        s = self.cell_size
        gp = self.pool_groups
        d = self.group_dim
        if n != gw * gw or c != self.hidden_size:
            raise ValueError(f"expected [B,{gw * gw},{self.hidden_size}], got {tuple(dz.shape)}")
        dz = dz.view(b, gw, gw, gp, d)
        beta = self._residual_write_weight(dz.dtype).permute(1, 2, 0)
        beta = beta[None, None, :, None, :, :, None]
        dx = dz[:, :, None, :, None, :, :] * beta
        return dx.reshape(b, hh, hh, c)

    def write_to_residual_add(self, dz: torch.Tensor, x_hi: torch.Tensor) -> torch.Tensor:
        return x_hi + self.write_to_residual(dz)

    def read(self, x: torch.Tensor) -> torch.Tensor:
        return self.read_from_residual(x)

    def write(self, dz: torch.Tensor) -> torch.Tensor:
        return self.write_to_residual(dz)


class ChannelwiseLocalTokenHyperConnectionCompilerFriendly(nn.Module):
    """Compiler-visible B/4 channel-wise read/write.

    This keeps the operation in normal PyTorch so TorchDynamo/Inductor can see
    through it. It specializes the common `pool_groups == hidden_size` case and
    avoids the extra singleton group dimension used by the generic version.
    """

    def __init__(
        self,
        high_grid: int,
        workspace_grid: int,
        hidden_size: int,
        pool_groups: int,
        use_softmax_write: bool = True,
        init_read: float = 1.0,
    ) -> None:
        super().__init__()
        if pool_groups != hidden_size:
            raise ValueError("CompilerFriendly LocalTHC requires pool_groups == hidden_size")
        if high_grid % workspace_grid != 0:
            raise ValueError(f"high_grid={high_grid} must be divisible by workspace_grid={workspace_grid}")
        self.high_grid = high_grid
        self.workspace_grid = workspace_grid
        self.hidden_size = hidden_size
        self.pool_groups = pool_groups
        self.group_dim = 1
        self.cell_size = high_grid // workspace_grid
        self.cell_tokens = self.cell_size * self.cell_size
        self.use_softmax_write = use_softmax_write
        self.read_logits = nn.Parameter(torch.zeros(hidden_size, self.cell_tokens))
        self.write_weight = nn.Parameter(torch.full((hidden_size, self.cell_tokens), float(init_read)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.constant_(self.read_logits, 0.0)
        nn.init.constant_(self.write_weight, 1.0)

    def _residual_read_weight(self, dtype: torch.dtype) -> torch.Tensor:
        if self.use_softmax_write:
            w = F.softmax(self.read_logits.float(), dim=-1)
        else:
            w = self.read_logits
        return w.to(dtype=dtype).view(self.hidden_size, self.cell_size, self.cell_size).permute(1, 2, 0)

    def _residual_write_weight(self, dtype: torch.dtype) -> torch.Tensor:
        return self.write_weight.to(dtype=dtype).view(self.hidden_size, self.cell_size, self.cell_size).permute(1, 2, 0)

    def read_from_residual(self, x: torch.Tensor) -> torch.Tensor:
        b, h, w, c = x.shape
        hh = self.high_grid
        gw = self.workspace_grid
        s = self.cell_size
        if h != hh or w != hh or c != self.hidden_size:
            raise ValueError(f"expected [B,{hh},{hh},{self.hidden_size}], got {tuple(x.shape)}")
        x_cells = x.view(b, gw, s, gw, s, c)
        alpha = self._residual_read_weight(x.dtype)
        z = (x_cells * alpha[None, None, :, None, :, :]).sum(dim=(2, 4))
        return z.reshape(b, gw * gw, c)

    def write_to_residual(self, dz: torch.Tensor) -> torch.Tensor:
        b, n, c = dz.shape
        hh = self.high_grid
        gw = self.workspace_grid
        s = self.cell_size
        if n != gw * gw or c != self.hidden_size:
            raise ValueError(f"expected [B,{gw * gw},{self.hidden_size}], got {tuple(dz.shape)}")
        beta = self._residual_write_weight(dz.dtype)
        dx = dz.view(b, gw, gw, c)[:, :, None, :, None, :] * beta[None, None, :, None, :, :]
        return dx.reshape(b, hh, hh, c)

    def write_to_residual_add(self, dz: torch.Tensor, x_hi: torch.Tensor) -> torch.Tensor:
        b, n, c = dz.shape
        hh = self.high_grid
        gw = self.workspace_grid
        s = self.cell_size
        if n != gw * gw or c != self.hidden_size:
            raise ValueError(f"expected [B,{gw * gw},{self.hidden_size}], got {tuple(dz.shape)}")
        beta = self._residual_write_weight(dz.dtype)
        dx = dz.view(b, gw, gw, c)[:, :, None, :, None, :] * beta[None, None, :, None, :, :]
        x_cells = x_hi.view(b, gw, s, gw, s, c)
        return (x_cells + dx).reshape(b, hh, hh, c)

    def read(self, x: torch.Tensor) -> torch.Tensor:
        return self.read_from_residual(x)

    def write(self, dz: torch.Tensor) -> torch.Tensor:
        return self.write_to_residual(dz)

    def write_add(self, dz: torch.Tensor, x_hi: torch.Tensor) -> torch.Tensor:
        return self.write_to_residual_add(dz, x_hi)


class ChannelwiseLocalTokenHyperConnectionTriton(nn.Module):
    """Triton implementation for B/4 channel-wise LocalTHC read/write.

    This intentionally supports only the current speed-critical shape:
    high_grid=64, workspace_grid=16, hidden_size=pool_groups.
    """

    def __init__(
        self,
        high_grid: int,
        workspace_grid: int,
        hidden_size: int,
        pool_groups: int,
        use_softmax_write: bool = True,
        init_read: float = 1.0,
    ) -> None:
        super().__init__()
        if high_grid != 64 or workspace_grid != 16:
            raise ValueError("Triton LocalTHC currently supports only high_grid=64, workspace_grid=16")
        if pool_groups != hidden_size:
            raise ValueError("Triton LocalTHC currently requires pool_groups == hidden_size")
        if not use_softmax_write:
            raise ValueError("Triton LocalTHC currently assumes softmax read weights")
        self.high_grid = high_grid
        self.workspace_grid = workspace_grid
        self.hidden_size = hidden_size
        self.pool_groups = pool_groups
        self.group_dim = 1
        self.cell_size = 4
        self.cell_tokens = 16
        self.use_softmax_write = use_softmax_write
        self.read_logits = nn.Parameter(torch.zeros(hidden_size, self.cell_tokens))
        self.write_weight = nn.Parameter(torch.full((hidden_size, self.cell_tokens), float(init_read)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.constant_(self.read_logits, 0.0)
        nn.init.constant_(self.write_weight, 1.0)

    def read_from_residual(self, x: torch.Tensor) -> torch.Tensor:
        alpha = F.softmax(self.read_logits.float(), dim=-1).to(dtype=x.dtype)
        return triton_local_write(x, alpha)

    def write_to_residual(self, dz: torch.Tensor) -> torch.Tensor:
        return triton_local_read(dz, self.write_weight.to(dtype=dz.dtype))

    def read(self, x: torch.Tensor) -> torch.Tensor:
        return self.read_from_residual(x)

    def write(self, dz: torch.Tensor) -> torch.Tensor:
        return self.write_to_residual(dz)


class ChannelwiseLocalTokenHyperConnectionTritonReadAdd(ChannelwiseLocalTokenHyperConnectionTriton):
    """Triton read/write with read and outer residual fused in forward."""

    def write_to_residual_add(self, dz: torch.Tensor, x_hi: torch.Tensor) -> torch.Tensor:
        return triton_local_read_add(dz, self.write_weight.to(dtype=dz.dtype), x_hi)

    def write_add(self, dz: torch.Tensor, x_hi: torch.Tensor) -> torch.Tensor:
        return self.write_to_residual_add(dz, x_hi)


class ChannelwiseLocalTokenHyperConnectionTritonFrozen(ChannelwiseLocalTokenHyperConnectionTriton):
    """Triton speed upper bound with no gradients for local read/write weights."""

    def read_from_residual(self, x: torch.Tensor) -> torch.Tensor:
        alpha = F.softmax(self.read_logits.float(), dim=-1).to(dtype=x.dtype)
        return triton_local_write_no_weight_grad(x, alpha)

    def write_to_residual_add(self, dz: torch.Tensor, x_hi: torch.Tensor) -> torch.Tensor:
        return triton_local_read_add_no_weight_grad(dz, self.write_weight.to(dtype=dz.dtype), x_hi)

    def write_to_residual(self, dz: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Use write_to_residual_add for TritonFrozen LocalTHC.")

    def read(self, x: torch.Tensor) -> torch.Tensor:
        return self.read_from_residual(x)

    def write_add(self, dz: torch.Tensor, x_hi: torch.Tensor) -> torch.Tensor:
        return self.write_to_residual_add(dz, x_hi)

    def write(self, dz: torch.Tensor) -> torch.Tensor:
        return self.write_to_residual(dz)


class ChannelwiseLocalTokenHyperConnectionTritonTraceable(ChannelwiseLocalTokenHyperConnectionTriton):
    """Triton read/write registered as compiler-visible torch.library ops."""

    def read_from_residual(self, x: torch.Tensor) -> torch.Tensor:
        alpha = F.softmax(self.read_logits.float(), dim=-1).to(dtype=x.dtype)
        return triton_local_write_traceable(x, alpha)

    def write_to_residual_add(self, dz: torch.Tensor, x_hi: torch.Tensor) -> torch.Tensor:
        return triton_local_read_add_traceable(dz, self.write_weight.to(dtype=dz.dtype), x_hi)

    def write_to_residual(self, dz: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Use write_to_residual_add for TritonTraceable LocalTHC.")

    def read(self, x: torch.Tensor) -> torch.Tensor:
        return self.read_from_residual(x)

    def write_add(self, dz: torch.Tensor, x_hi: torch.Tensor) -> torch.Tensor:
        return self.write_to_residual_add(dz, x_hi)

    def write(self, dz: torch.Tensor) -> torch.Tensor:
        return self.write_to_residual(dz)


class ChannelwiseLocalTokenHyperConnectionDepthwiseConv(nn.Module):
    """Depthwise-conv implementation of channel-wise LocalTHC read/write.

    For B/4, pool_groups == hidden_size and each channel has its own 4x4 local
    read/write weights. That is equivalent to depthwise Conv2d and
    ConvTranspose2d with stride=cell_size, but avoids large NHWC broadcast
    tensors and keeps Conv2d gradient layouts standard for DDP.
    """

    def __init__(
        self,
        high_grid: int,
        workspace_grid: int,
        hidden_size: int,
        pool_groups: int,
        use_softmax_write: bool = True,
        init_read: float = 1.0,
    ) -> None:
        super().__init__()
        if high_grid % workspace_grid != 0:
            raise ValueError(f"high_grid={high_grid} must be divisible by workspace_grid={workspace_grid}")
        if pool_groups != hidden_size:
            raise ValueError("DepthwiseConv LocalTHC currently requires pool_groups == hidden_size")
        self.high_grid = high_grid
        self.workspace_grid = workspace_grid
        self.hidden_size = hidden_size
        self.pool_groups = pool_groups
        self.cell_size = high_grid // workspace_grid
        self.cell_tokens = self.cell_size * self.cell_size
        self.use_softmax_write = use_softmax_write
        self.read_logits = nn.Parameter(torch.zeros(hidden_size, self.cell_tokens))
        self.write_weight = nn.Parameter(torch.full((hidden_size, self.cell_tokens), float(init_read)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.constant_(self.read_logits, 0.0)
        nn.init.constant_(self.write_weight, 1.0)

    def _read_kernel(self, dtype: torch.dtype) -> torch.Tensor:
        if self.use_softmax_write:
            w = F.softmax(self.read_logits.float(), dim=-1)
        else:
            w = self.read_logits
        return w.to(dtype=dtype).view(self.hidden_size, 1, self.cell_size, self.cell_size)

    def _write_kernel(self, dtype: torch.dtype) -> torch.Tensor:
        return self.write_weight.to(dtype=dtype).view(self.hidden_size, 1, self.cell_size, self.cell_size)

    def read_from_residual(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        if c != self.hidden_size or h != self.high_grid or w != self.high_grid:
            raise ValueError(f"expected [B,{self.hidden_size},{self.high_grid},{self.high_grid}], got {tuple(x.shape)}")
        z = F.conv2d(
            x,
            self._read_kernel(x.dtype),
            bias=None,
            stride=self.cell_size,
            groups=self.hidden_size,
        )
        return z.flatten(2).transpose(1, 2)

    def write_to_residual(self, dz: torch.Tensor) -> torch.Tensor:
        b, n, c = dz.shape
        if n != self.workspace_grid * self.workspace_grid or c != self.hidden_size:
            raise ValueError(f"expected [B,{self.workspace_grid * self.workspace_grid},{self.hidden_size}], got {tuple(dz.shape)}")
        dz = dz.transpose(1, 2).reshape(b, c, self.workspace_grid, self.workspace_grid)
        return F.conv_transpose2d(
            dz,
            self._write_kernel(dz.dtype),
            bias=None,
            stride=self.cell_size,
            groups=self.hidden_size,
        )

    def read(self, x: torch.Tensor) -> torch.Tensor:
        return self.read_from_residual(x)

    def write(self, dz: torch.Tensor) -> torch.Tensor:
        return self.write_to_residual(dz)


class ChannelwiseLocalTokenHyperConnectionFixedPool(nn.Module):
    """Fixed avg-read / broadcast-write speed reference."""

    def __init__(self, high_grid: int, workspace_grid: int, hidden_size: int, pool_groups: int) -> None:
        super().__init__()
        if high_grid % workspace_grid != 0:
            raise ValueError(f"high_grid={high_grid} must be divisible by workspace_grid={workspace_grid}")
        self.high_grid = high_grid
        self.workspace_grid = workspace_grid
        self.hidden_size = hidden_size
        self.cell_size = high_grid // workspace_grid

    def read_from_residual(self, x: torch.Tensor) -> torch.Tensor:
        z = F.avg_pool2d(x, kernel_size=self.cell_size, stride=self.cell_size)
        return z.flatten(2).transpose(1, 2)

    def write_to_residual(self, dz: torch.Tensor) -> torch.Tensor:
        b, n, c = dz.shape
        if n != self.workspace_grid * self.workspace_grid or c != self.hidden_size:
            raise ValueError(f"expected [B,{self.workspace_grid * self.workspace_grid},{self.hidden_size}], got {tuple(dz.shape)}")
        dz = dz.transpose(1, 2).reshape(b, c, self.workspace_grid, self.workspace_grid)
        return dz.repeat_interleave(self.cell_size, dim=2).repeat_interleave(self.cell_size, dim=3)

    def read(self, x: torch.Tensor) -> torch.Tensor:
        return self.read_from_residual(x)

    def write(self, dz: torch.Tensor) -> torch.Tensor:
        return self.write_to_residual(dz)


class LocalTHCJiTBlock(nn.Module):
    """Normal-AdaLN local token hyper-connection block."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        high_grid: int,
        workspace_grid: int = 16,
        pool_groups: int | None = None,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        attn_backend: str = "flash",
    ) -> None:
        super().__init__()
        pool_groups = hidden_size if pool_groups is None else pool_groups
        self.hyper = ChannelwiseLocalTokenHyperConnection(high_grid, workspace_grid, hidden_size, pool_groups)
        self.branch = WorkspaceJiTBranch(hidden_size, num_heads, mlp_ratio, attn_drop, proj_drop, attn_backend)

    def forward(self, x_hi: torch.Tensor, c: torch.Tensor, rope: VisionRotaryEmbeddingFast) -> torch.Tensor:
        z = self.hyper.read_from_residual(x_hi)
        dz = self.branch(z, c, rope)
        if hasattr(self.hyper, "write_to_residual_add"):
            return self.hyper.write_to_residual_add(dz, x_hi)
        return x_hi + self.hyper.write_to_residual(dz)


class LocalTHCSharedAdaLNJiTBlock(nn.Module):
    """Shared-AdaLN local token hyper-connection block."""

    hyper_cls = ChannelwiseLocalTokenHyperConnection

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        high_grid: int,
        workspace_grid: int = 16,
        pool_groups: int | None = None,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        attn_backend: str = "flash",
    ) -> None:
        super().__init__()
        pool_groups = hidden_size if pool_groups is None else pool_groups
        self.hyper = self.hyper_cls(high_grid, workspace_grid, hidden_size, pool_groups)
        self.branch = WorkspaceSharedAdaLNBranch(hidden_size, num_heads, mlp_ratio, attn_drop, proj_drop, attn_backend)

    def forward(self, x_hi: torch.Tensor, t6: torch.Tensor, rope: VisionRotaryEmbeddingFast) -> torch.Tensor:
        z = self.hyper.read_from_residual(x_hi)
        dz = self.branch(z, t6, rope)
        if hasattr(self.hyper, "write_to_residual_add"):
            return self.hyper.write_to_residual_add(dz, x_hi)
        return x_hi + self.hyper.write_to_residual(dz)


class LocalTHCNCHWSharedAdaLNJiTBlock(nn.Module):
    """Shared-AdaLN LocalTHC block with NCHW persistent stream."""

    hyper_cls = ChannelwiseLocalTokenHyperConnectionDepthwiseConv

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        high_grid: int,
        workspace_grid: int = 16,
        pool_groups: int | None = None,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        attn_backend: str = "flash",
    ) -> None:
        super().__init__()
        pool_groups = hidden_size if pool_groups is None else pool_groups
        self.hyper = self.hyper_cls(high_grid, workspace_grid, hidden_size, pool_groups)
        self.branch = WorkspaceSharedAdaLNBranch(hidden_size, num_heads, mlp_ratio, attn_drop, proj_drop, attn_backend)

    def forward(self, x_hi: torch.Tensor, t6: torch.Tensor, rope: VisionRotaryEmbeddingFast) -> torch.Tensor:
        z = self.hyper.read_from_residual(x_hi)
        dz = self.branch(z, t6, rope)
        return x_hi + self.hyper.write_to_residual(dz)


class LocalTHCNCHWFixedPoolSharedAdaLNJiTBlock(LocalTHCNCHWSharedAdaLNJiTBlock):
    """NCHW LocalTHC block using fixed pooling for speed diagnosis."""

    hyper_cls = ChannelwiseLocalTokenHyperConnectionFixedPool


class LocalTHCTritonSharedAdaLNJiTBlock(LocalTHCSharedAdaLNJiTBlock):
    """NHWC LocalTHC block with Triton read/write kernels."""

    hyper_cls = ChannelwiseLocalTokenHyperConnectionTriton


class LocalTHCTritonReadAddSharedAdaLNJiTBlock(LocalTHCSharedAdaLNJiTBlock):
    """Triton read/write with fused read+residual-add forward."""

    hyper_cls = ChannelwiseLocalTokenHyperConnectionTritonReadAdd


class LocalTHCTritonFrozenSharedAdaLNJiTBlock(LocalTHCSharedAdaLNJiTBlock):
    """Triton speed upper-bound block with frozen local read/write weights."""

    hyper_cls = ChannelwiseLocalTokenHyperConnectionTritonFrozen


class LocalTHCTritonTraceableSharedAdaLNJiTBlock(LocalTHCSharedAdaLNJiTBlock):
    """Triton read/write via torch.library.triton_op."""

    hyper_cls = ChannelwiseLocalTokenHyperConnectionTritonTraceable


class LocalTHCCompilerFriendlySharedAdaLNJiTBlock(LocalTHCSharedAdaLNJiTBlock):
    """Shared-AdaLN block with compiler-visible channel-wise read/write."""

    hyper_cls = ChannelwiseLocalTokenHyperConnectionCompilerFriendly


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int) -> None:
        super().__init__()
        self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        return self.linear(modulate(self.norm_final(x), shift, scale))


class FinalLayerSharedAdaLN(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int) -> None:
        super().__init__()
        self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)

    def forward(self, x: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        shift, scale = t2.chunk(2, dim=1)
        return self.linear(modulate(self.norm_final(x), shift, scale))


class LocalTHCJiT(nn.Module):
    """Local Token Hyper-Connection JiT with per-block AdaLN."""

    block_cls = LocalTHCJiTBlock
    final_cls = FinalLayer
    x_embedder_cls = BottleneckPatchEmbedNHWC

    def __init__(
        self,
        input_size: int = 256,
        patch_size: int = 4,
        in_channels: int = 3,
        hidden_size: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        num_classes: int = 1000,
        bottleneck_dim: int = 128,
        workspace_grid: int = 16,
        pool_groups: int | None = None,
        attn_backend: str = "flash",
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.workspace_grid = workspace_grid
        high_grid = input_size // patch_size
        if high_grid % workspace_grid != 0:
            raise ValueError(f"input_size/patch_size={high_grid} must be divisible by workspace_grid={workspace_grid}")
        self.high_grid = high_grid
        self.num_high_tokens = high_grid * high_grid
        self.num_workspace_tokens = workspace_grid * workspace_grid
        self.pool_groups = hidden_size if pool_groups is None else pool_groups

        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size)
        self.x_embedder = self.x_embedder_cls(input_size, patch_size, in_channels, bottleneck_dim, hidden_size)
        half_head_dim = hidden_size // num_heads // 2
        self.workspace_rope = VisionRotaryEmbeddingFast(half_head_dim, workspace_grid, num_prefix_tokens=0)
        self.blocks = nn.ModuleList([
            self.block_cls(
                hidden_size=hidden_size,
                num_heads=num_heads,
                high_grid=high_grid,
                workspace_grid=workspace_grid,
                pool_groups=self.pool_groups,
                mlp_ratio=mlp_ratio,
                attn_drop=attn_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0,
                proj_drop=proj_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0,
                attn_backend=attn_backend,
            )
            for i in range(depth)
        ])
        self.final_layer = self.final_cls(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        state_dict = migrate_lthc_state_dict_keys(state_dict)
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def initialize_weights(self) -> None:
        def basic_init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(basic_init)
        if hasattr(self.x_embedder, "proj"):
            nn.init.xavier_uniform_(self.x_embedder.proj.weight.data.view(self.x_embedder.proj.weight.shape[0], -1))
            if self.x_embedder.proj.bias is not None:
                nn.init.constant_(self.x_embedder.proj.bias, 0)
        else:
            nn.init.xavier_uniform_(self.x_embedder.proj1.weight.data.view(self.x_embedder.proj1.weight.shape[0], -1))
            nn.init.xavier_uniform_(self.x_embedder.proj2.weight.data.view(self.x_embedder.proj2.weight.shape[0], -1))
            nn.init.constant_(self.x_embedder.proj2.bias, 0)
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        for block in self.blocks:
            if hasattr(block.branch, "adaLN_modulation"):
                nn.init.constant_(block.branch.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.branch.adaLN_modulation[-1].bias, 0)
        if hasattr(self.final_layer, "adaLN_modulation"):
            nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        c = self.out_channels
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        if h * w != x.shape[1]:
            raise ValueError(f"token count {x.shape[1]} is not square")
        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = x.permute(0, 5, 1, 3, 2, 4)
        return x.reshape(x.shape[0], c, h * p, w * p)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        c = self.t_embedder(t) + self.y_embedder(y)
        x_hi = self.x_embedder(x)
        for block in self.blocks:
            x_hi = block(x_hi, c, self.workspace_rope)
        b, hh, wh, c_dim = x_hi.shape
        x_out = self.final_layer(x_hi.reshape(b, hh * wh, c_dim), c)
        return self.unpatchify(x_out)


class LocalTHCSharedAdaLNJiT(LocalTHCJiT):
    """Local Token Hyper-Connection JiT with shared AdaLN heads."""

    block_cls = LocalTHCSharedAdaLNJiTBlock
    final_cls = FinalLayerSharedAdaLN

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.shared_adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(self.hidden_size, 6 * self.hidden_size, bias=True))
        self.shared_final_modulation = nn.Sequential(nn.SiLU(), nn.Linear(self.hidden_size, 2 * self.hidden_size, bias=True))
        self._init_shared_adaln()

    def _init_shared_adaln(self) -> None:
        for module in (self.shared_adaLN_modulation[-1], self.shared_final_modulation[-1]):
            nn.init.constant_(module.weight, 0)
            nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        c = self.t_embedder(t) + self.y_embedder(y)
        t6 = self.shared_adaLN_modulation(c)
        t2 = self.shared_final_modulation(c)
        x_hi = self.x_embedder(x)
        for block in self.blocks:
            x_hi = block(x_hi, t6, self.workspace_rope)
        b, hh, wh, c_dim = x_hi.shape
        x_out = self.final_layer(x_hi.reshape(b, hh * wh, c_dim), t2)
        return self.unpatchify(x_out)


class LocalTHCInContextSharedAdaLNJiT(LocalTHCSharedAdaLNJiT):
    """LocalTHC with JiT-style time AdaLN and in-context class prefix tokens.

    This variant keeps the LocalTHC high-resolution residual interface, but
    aligns the workspace branch conditioning with ``JiTSharedTimeAdaLN``:
    timestep information enters only through shared AdaLN, while class
    information enters attention as prefix tokens starting at
    ``class_token_start``. The prefix tokens are persistent across subsequent
    workspace blocks; only image workspace updates are written back to the
    high-resolution residual grid.
    """

    def __init__(
        self,
        *args,
        num_class_tokens: int = 32,
        class_token_start: int = 4,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.num_class_tokens = num_class_tokens
        self.class_token_start = class_token_start
        grid = self.high_grid
        pos = get_2d_sincos_pos_embed(self.hidden_size, grid)
        self.register_buffer(
            "pos_embed",
            torch.from_numpy(pos).float().view(1, grid, grid, self.hidden_size),
            persistent=False,
        )
        self.class_token_pos = nn.Parameter(torch.randn(1, num_class_tokens, self.hidden_size) * 0.02)
        half_head_dim = self.hidden_size // self.num_heads // 2
        self.workspace_rope_incontext = VisionRotaryEmbeddingFast(
            half_head_dim,
            self.workspace_grid,
            num_prefix_tokens=num_class_tokens,
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        t_emb = self.t_embedder(t)
        y_emb = self.y_embedder(y)
        t6 = self.shared_adaLN_modulation(t_emb)
        t2 = self.shared_final_modulation(t_emb)
        x_hi = self.x_embedder(x)
        x_hi = x_hi + self.pos_embed.to(device=x.device, dtype=x_hi.dtype)
        class_tokens = y_emb.unsqueeze(1).expand(-1, self.num_class_tokens, -1) + self.class_token_pos.to(
            device=x.device,
            dtype=y_emb.dtype,
        )

        for i, block in enumerate(self.blocks):
            z = block.hyper.read_from_residual(x_hi)
            if self.num_class_tokens > 0 and i >= self.class_token_start:
                seq = torch.cat([class_tokens, z], dim=1)
                dseq = block.branch(seq, t6, self.workspace_rope_incontext)
                class_tokens = class_tokens + dseq[:, : self.num_class_tokens]
                dz = dseq[:, self.num_class_tokens :]
            else:
                dz = block.branch(z, t6, self.workspace_rope)
            if hasattr(block.hyper, "write_to_residual_add"):
                x_hi = block.hyper.write_to_residual_add(dz, x_hi)
            else:
                x_hi = x_hi + block.hyper.write_to_residual(dz)

        b, hh, wh, c_dim = x_hi.shape
        x_out = self.final_layer(x_hi.reshape(b, hh * wh, c_dim), t2)
        return self.unpatchify(x_out)


class LocalTHCInContextCompilerFriendlySharedAdaLNJiT(LocalTHCInContextSharedAdaLNJiT):
    """In-context B/8 path with compiler-friendly local read/write-add.

    This keeps the architecture identical to ``LocalTHCInContextSharedAdaLNJiT``
    but replaces the generic LocalTHC interface with the per-channel
    compiler-visible implementation. For B/8 this removes the unused singleton
    group dimension and exposes the local residual write-add as one expression
    that Inductor can fuse more easily.
    """

    block_cls = LocalTHCCompilerFriendlySharedAdaLNJiTBlock


class LocalTHCDirectSharedAdaLNJiT(LocalTHCSharedAdaLNJiT):
    """LocalTHC shared-AdaLN with direct 3->hidden patch embedding."""

    x_embedder_cls = DirectPatchEmbedNHWC


class LocalTHCTritonSharedAdaLNJiT(LocalTHCSharedAdaLNJiT):
    """Shared-AdaLN LocalTHC with Triton B/4 read/write kernels."""

    block_cls = LocalTHCTritonSharedAdaLNJiTBlock


class LocalTHCDirectTritonSharedAdaLNJiT(LocalTHCTritonSharedAdaLNJiT):
    """Direct patch embed + Triton B/4 read/write kernels."""

    x_embedder_cls = DirectPatchEmbedNHWC


class LocalTHCTritonReadAddSharedAdaLNJiT(LocalTHCSharedAdaLNJiT):
    """Shared-AdaLN LocalTHC with legacy-named Triton fused write_add."""

    block_cls = LocalTHCTritonReadAddSharedAdaLNJiTBlock


class LocalTHCDirectTritonReadAddSharedAdaLNJiT(LocalTHCTritonReadAddSharedAdaLNJiT):
    """Direct patch embed + legacy-named Triton fused write_add."""

    x_embedder_cls = DirectPatchEmbedNHWC


class LocalTHCTritonFrozenSharedAdaLNJiT(LocalTHCSharedAdaLNJiT):
    """Shared-AdaLN LocalTHC speed upper-bound with frozen local weights."""

    block_cls = LocalTHCTritonFrozenSharedAdaLNJiTBlock


class LocalTHCDirectTritonFrozenSharedAdaLNJiT(LocalTHCTritonFrozenSharedAdaLNJiT):
    """Direct patch embed + frozen-weight Triton read/write upper-bound."""

    x_embedder_cls = DirectPatchEmbedNHWC


class LocalTHCTritonTraceableSharedAdaLNJiT(LocalTHCSharedAdaLNJiT):
    """Shared-AdaLN LocalTHC with compiler-visible Triton ops."""

    block_cls = LocalTHCTritonTraceableSharedAdaLNJiTBlock


class LocalTHCDirectTritonTraceableSharedAdaLNJiT(LocalTHCTritonTraceableSharedAdaLNJiT):
    """Direct patch embed + compiler-visible Triton ops."""

    x_embedder_cls = DirectPatchEmbedNHWC


class LocalTHCCompilerFriendlySharedAdaLNJiT(LocalTHCSharedAdaLNJiT):
    """Shared-AdaLN LocalTHC with PyTorch compiler-friendly read/write."""

    block_cls = LocalTHCCompilerFriendlySharedAdaLNJiTBlock


class LocalTHCDirectCompilerFriendlySharedAdaLNJiT(LocalTHCCompilerFriendlySharedAdaLNJiT):
    """Direct patch embed + compiler-friendly read/write."""

    x_embedder_cls = DirectPatchEmbedNHWC


class SharedLocalRead(nn.Module):
    """Depth-shared local read operator for lazy LocalTHC execution.

    The operator maps a high-resolution NHWC residual stream to the workspace:
    [B, Hh, Hh, C] -> [B, Gw*Gw, C]. Unlike the normal LocalTHC block-local
    hyper-connection, this read weight is owned by the top-level model and is
    shared across all layers.
    """

    def __init__(
        self,
        high_grid: int,
        workspace_grid: int,
        hidden_size: int,
        pool_groups: int,
        use_softmax_write: bool = True,
    ) -> None:
        super().__init__()
        if high_grid % workspace_grid != 0:
            raise ValueError(f"high_grid={high_grid} must be divisible by workspace_grid={workspace_grid}")
        if hidden_size % pool_groups != 0:
            raise ValueError(f"hidden_size={hidden_size} must be divisible by pool_groups={pool_groups}")
        self.high_grid = high_grid
        self.workspace_grid = workspace_grid
        self.hidden_size = hidden_size
        self.pool_groups = pool_groups
        self.group_dim = hidden_size // pool_groups
        self.cell_size = high_grid // workspace_grid
        self.cell_tokens = self.cell_size * self.cell_size
        self.use_softmax_write = use_softmax_write
        self.read_logits = nn.Parameter(torch.zeros(pool_groups, self.cell_tokens))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.constant_(self.read_logits, 0.0)

    def alpha(self, dtype: torch.dtype | None = None) -> torch.Tensor:
        if self.use_softmax_write:
            alpha = F.softmax(self.read_logits.float(), dim=-1)
        else:
            alpha = self.read_logits
        return alpha if dtype is None else alpha.to(dtype=dtype)

    def read(self, x: torch.Tensor) -> torch.Tensor:
        b, h, w, c = x.shape
        hh = self.high_grid
        gw = self.workspace_grid
        s = self.cell_size
        gp = self.pool_groups
        d = self.group_dim
        if h != hh or w != hh or c != self.hidden_size:
            raise ValueError(f"expected [B,{hh},{hh},{self.hidden_size}], got {tuple(x.shape)}")
        x = x.view(b, gw, s, gw, s, gp, d)
        alpha = self.alpha(dtype=x.dtype).view(gp, s, s).permute(1, 2, 0)
        z = (x * alpha[None, None, :, None, :, :, None]).sum(dim=(2, 4))
        return z.reshape(b, gw * gw, c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.read(x)


class LayerLocalWrite(nn.Module):
    """Layer-specific local write operator used with a shared read."""

    def __init__(
        self,
        high_grid: int,
        workspace_grid: int,
        hidden_size: int,
        pool_groups: int,
        init_read: float = 1.0,
    ) -> None:
        super().__init__()
        if high_grid % workspace_grid != 0:
            raise ValueError(f"high_grid={high_grid} must be divisible by workspace_grid={workspace_grid}")
        if hidden_size % pool_groups != 0:
            raise ValueError(f"hidden_size={hidden_size} must be divisible by pool_groups={pool_groups}")
        self.high_grid = high_grid
        self.workspace_grid = workspace_grid
        self.hidden_size = hidden_size
        self.pool_groups = pool_groups
        self.group_dim = hidden_size // pool_groups
        self.cell_size = high_grid // workspace_grid
        self.cell_tokens = self.cell_size * self.cell_size
        self.write_weight = nn.Parameter(torch.full((pool_groups, self.cell_tokens), float(init_read)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.constant_(self.write_weight, 1.0)

    def beta(self, dtype: torch.dtype | None = None) -> torch.Tensor:
        return self.write_weight if dtype is None else self.write_weight.to(dtype=dtype)

    def write(self, dz: torch.Tensor) -> torch.Tensor:
        b, n, c = dz.shape
        hh = self.high_grid
        gw = self.workspace_grid
        s = self.cell_size
        gp = self.pool_groups
        d = self.group_dim
        if n != gw * gw or c != self.hidden_size:
            raise ValueError(f"expected [B,{gw * gw},{self.hidden_size}], got {tuple(dz.shape)}")
        dz = dz.view(b, gw, gw, gp, d)
        beta = self.beta(dtype=dz.dtype).view(gp, s, s).permute(1, 2, 0)
        dx = dz[:, :, None, :, None, :, :] * beta[None, None, :, None, :, :, None]
        return dx.reshape(b, hh, hh, c)

    def write_add(self, x_hi: torch.Tensor, dz: torch.Tensor) -> torch.Tensor:
        return x_hi + self.write(dz)

    def gamma_from_alpha(self, alpha: torch.Tensor) -> torch.Tensor:
        beta = self.beta(dtype=alpha.dtype)
        gamma_g = (alpha * beta).sum(dim=-1)
        if self.group_dim == 1:
            return gamma_g
        return gamma_g[:, None].expand(self.pool_groups, self.group_dim).reshape(self.hidden_size)


class SharedReadLocalTHCSharedAdaLNJiTBlock(nn.Module):
    """LocalTHC block using a depth-shared read and layer-specific write."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        high_grid: int,
        workspace_grid: int = 16,
        pool_groups: int | None = None,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        attn_backend: str = "flash",
    ) -> None:
        super().__init__()
        pool_groups = hidden_size if pool_groups is None else pool_groups
        self.write = LayerLocalWrite(high_grid, workspace_grid, hidden_size, pool_groups)
        self.branch = WorkspaceSharedAdaLNBranch(hidden_size, num_heads, mlp_ratio, attn_drop, proj_drop, attn_backend)

    @property
    def read(self) -> LayerLocalWrite:
        """Backward-compatible module alias for old research checkpoints."""
        return self.write

    def forward_naive(
        self,
        x_hi: torch.Tensor,
        t6: torch.Tensor,
        rope: VisionRotaryEmbeddingFast,
        shared_read: SharedLocalRead,
    ) -> torch.Tensor:
        z = shared_read.read(x_hi)
        dz = self.branch(z, t6, rope)
        return self.write.write_add(x_hi, dz)

    def branch_update(self, z: torch.Tensor, t6: torch.Tensor, rope: VisionRotaryEmbeddingFast) -> torch.Tensor:
        return self.branch(z, t6, rope)

    def lazy_workspace_update(self, z: torch.Tensor, dz: torch.Tensor, shared_read: SharedLocalRead) -> torch.Tensor:
        alpha = shared_read.alpha(dtype=dz.dtype)
        gamma = self.write.gamma_from_alpha(alpha)
        return z + gamma[None, None, :] * dz


class SharedReadLocalTHCSharedAdaLNJiT(LocalTHCSharedAdaLNJiT):
    """Shared-read LocalTHC with shared AdaLN conditioning.

    The lazy path is exact for the local channel/group read-write operators:
    once the read R is shared across layers, the workspace can be updated as
    z_{l+1} = z_l + (R P_l) dz_l, where R P_l is only a per-channel/group scale.
    """

    block_cls = SharedReadLocalTHCSharedAdaLNJiTBlock

    def __init__(self, *args, use_lazy: bool = True, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.shared_read = SharedLocalRead(
            high_grid=self.high_grid,
            workspace_grid=self.workspace_grid,
            hidden_size=self.hidden_size,
            pool_groups=self.pool_groups,
            use_softmax_write=True,
        )
        self.use_lazy = bool(use_lazy)

    @property
    def shared_write(self) -> SharedLocalRead:
        """Backward-compatible module alias for old research code."""
        return self.shared_read

    def forward_naive(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        c = self.t_embedder(t) + self.y_embedder(y)
        t6 = self.shared_adaLN_modulation(c)
        t2 = self.shared_final_modulation(c)
        x_hi = self.x_embedder(x)
        for block in self.blocks:
            x_hi = block.forward_naive(x_hi, t6, self.workspace_rope, self.shared_read)
        b, hh, wh, c_dim = x_hi.shape
        x_out = self.final_layer(x_hi.reshape(b, hh * wh, c_dim), t2)
        return self.unpatchify(x_out)

    def final_accumulate_reference(self, x0: torch.Tensor, dz_list: list[torch.Tensor]) -> torch.Tensor:
        x_hi = x0
        for block, dz in zip(self.blocks, dz_list, strict=True):
            x_hi = block.write.write_add(x_hi, dz)
        return x_hi

    def forward_lazy(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        c = self.t_embedder(t) + self.y_embedder(y)
        t6 = self.shared_adaLN_modulation(c)
        t2 = self.shared_final_modulation(c)
        x0 = self.x_embedder(x)
        z = self.shared_read.read(x0)
        dz_list = []
        for block in self.blocks:
            dz = block.branch_update(z, t6, self.workspace_rope)
            dz_list.append(dz)
            z = block.lazy_workspace_update(z, dz, self.shared_read)
        x_hi = self.final_accumulate_reference(x0, dz_list)
        b, hh, wh, c_dim = x_hi.shape
        x_out = self.final_layer(x_hi.reshape(b, hh * wh, c_dim), t2)
        return self.unpatchify(x_out)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor, use_lazy: bool | None = None) -> torch.Tensor:
        use_lazy = self.use_lazy if use_lazy is None else use_lazy
        if use_lazy:
            return self.forward_lazy(x, t, y)
        return self.forward_naive(x, t, y)


class SharedReadFusedFinalLocalTHCSharedAdaLNJiT(SharedReadLocalTHCSharedAdaLNJiT):
    """Shared-read LocalTHC using a fused final high-res accumulate kernel.

    This keeps the same workspace recurrence as ``SharedReadLocalTHCSharedAdaLNJiT``
    but replaces the terminal Python loop over 12 ``write_add`` calls with one
    fixed-shape Triton op for B/4. The stack operations are intentionally kept
    explicit so we can test whether the fused high-res pass helps under
    ``torch.compile`` before writing a more invasive no-stack variant.
    """

    def final_accumulate_reference(self, x0: torch.Tensor, dz_list: list[torch.Tensor]) -> torch.Tensor:
        dz_stack = torch.stack(dz_list, dim=0).contiguous()
        beta_stack = torch.stack([block.write.write_weight for block in self.blocks], dim=0).contiguous()
        return triton_final_accumulate_b4_traceable(x0, dz_stack, beta_stack)


class SharedReadFusedFinal12LocalTHCSharedAdaLNJiT(SharedReadLocalTHCSharedAdaLNJiT):
    """No-stack fused-final shared-read LocalTHC for fixed depth=12 B/4."""

    def final_accumulate_reference(self, x0: torch.Tensor, dz_list: list[torch.Tensor]) -> torch.Tensor:
        if len(dz_list) != 12 or len(self.blocks) != 12:
            raise ValueError("SharedReadFusedFinal12LocalTHCSharedAdaLNJiT requires depth=12")
        betas = [block.write.write_weight for block in self.blocks]
        return triton_final_accumulate_b4_12_traceable(x0, *dz_list, *betas)


# Backward-compatible class aliases. The old "SharedWrite" name referred to
# the same high-res -> workspace operator that the README now correctly calls
# "read".
SharedLocalWrite = SharedLocalRead
LayerLocalRead = LayerLocalWrite
SharedWriteLocalTHCSharedAdaLNJiTBlock = SharedReadLocalTHCSharedAdaLNJiTBlock
SharedWriteLocalTHCSharedAdaLNJiT = SharedReadLocalTHCSharedAdaLNJiT
SharedWriteFusedFinalLocalTHCSharedAdaLNJiT = SharedReadFusedFinalLocalTHCSharedAdaLNJiT
SharedWriteFusedFinal12LocalTHCSharedAdaLNJiT = SharedReadFusedFinal12LocalTHCSharedAdaLNJiT


class TriangularLazyFusedFinal12LocalTHCDirectTritonSharedAdaLNJiT(LocalTHCDirectTritonTraceableSharedAdaLNJiT):
    """Unshared local residual-read with exact triangular lazy execution.

    Each layer keeps its own local residual-read ``R_l`` and residual-write
    ``P_l``. Because both maps are channelwise linear inside each 4x4 local
    cell, ``R_l P_j`` collapses to a per-channel scale:

        gamma[l, j, c] = sum_r alpha[l, c, r] * beta[j, c, r].

    Therefore the high-resolution residual stream does not need to be
    materialized between layers. The workspace input to layer ``l`` is:

        z_l = R_l x0 + sum_{j < l} gamma[l, j] * dz_j.

    The final high-resolution stream is still:

        x_L = x0 + sum_j P_j dz_j,

    computed with the existing fused final-accumulate kernel.
    """

    def _expanded_alpha(self, block: nn.Module, dtype: torch.dtype) -> torch.Tensor:
        hyper = block.hyper
        if hyper.use_softmax_write:
            alpha = F.softmax(hyper.read_logits.float(), dim=-1)
        else:
            alpha = hyper.read_logits
        alpha = alpha.to(dtype=dtype)
        if hyper.group_dim == 1:
            return alpha
        return alpha[:, None, :].expand(hyper.pool_groups, hyper.group_dim, hyper.cell_tokens).reshape(
            hyper.hidden_size, hyper.cell_tokens
        )

    def _expanded_beta(self, block: nn.Module, dtype: torch.dtype) -> torch.Tensor:
        hyper = block.hyper
        beta = hyper.write_weight.to(dtype=dtype)
        if hyper.group_dim == 1:
            return beta
        return beta[:, None, :].expand(hyper.pool_groups, hyper.group_dim, hyper.cell_tokens).reshape(
            hyper.hidden_size, hyper.cell_tokens
        )

    def final_accumulate_reference(self, x0: torch.Tensor, dz_list: list[torch.Tensor]) -> torch.Tensor:
        if len(dz_list) != 12 or len(self.blocks) != 12:
            raise ValueError("TriangularLazyFusedFinal12LocalTHCDirectTritonSharedAdaLNJiT requires depth=12")
        betas = [self._expanded_beta(block, dz_list[0].dtype).contiguous() for block in self.blocks]
        return triton_final_accumulate_b4_12_traceable(x0, *dz_list, *betas)

    def forward_naive(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Reference high-resolution recurrence for correctness checks."""
        return super().forward(x, t, y)

    def forward_lazy(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        c = self.t_embedder(t) + self.y_embedder(y)
        t6 = self.shared_adaLN_modulation(c)
        t2 = self.shared_final_modulation(c)
        x0 = self.x_embedder(x)

        alpha_stack = torch.stack([self._expanded_alpha(block, x0.dtype) for block in self.blocks], dim=0)
        beta_stack = torch.stack([self._expanded_beta(block, x0.dtype) for block in self.blocks], dim=0)
        base_stack = triton_all_base_b4_12_traceable(x0, alpha_stack.contiguous())
        gamma = (alpha_stack[:, None, :, :] * beta_stack[None, :, :, :]).sum(dim=-1)
        zero_dz = torch.zeros_like(base_stack[0])

        dz_list: list[torch.Tensor] = []
        for layer_idx, block in enumerate(self.blocks):
            z = base_stack[layer_idx]
            if dz_list:
                dz_args = tuple(dz_list[j] if j < layer_idx else zero_dz for j in range(12))
                z = triton_triangular_active_ops[layer_idx - 1](
                    z,
                    *dz_args,
                    gamma[layer_idx].contiguous(),
                )
            dz = block.branch(z, t6, self.workspace_rope)
            dz_list.append(dz)

        x_hi = self.final_accumulate_reference(x0, dz_list)
        b, hh, wh, c_dim = x_hi.shape
        x_out = self.final_layer(x_hi.reshape(b, hh * wh, c_dim), t2)
        return self.unpatchify(x_out)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor, use_lazy: bool = True) -> torch.Tensor:
        if use_lazy:
            return self.forward_lazy(x, t, y)
        return self.forward_naive(x, t, y)


class TriangularLazyFusedFinal12LocalTHCTritonTraceableSharedAdaLNJiT(
    TriangularLazyFusedFinal12LocalTHCDirectTritonSharedAdaLNJiT
):
    """Non-shared LocalTHC triangular lazy path with bottleneck patch embed.

    This is the faithful B/4 LocalTHC architecture used by
    ``local_thc_jit_shared_adaln_b4``: per-layer local residual reads/writes,
    shared AdaLN, and the original two-stage bottleneck patch embed. The only
    change is the exact lazy/fused execution of the local interface.
    """

    x_embedder_cls = BottleneckPatchEmbedNHWC


class LocalTHCNCHWSharedAdaLNJiT(LocalTHCJiT):
    """Optimized LocalTHC shared-AdaLN model.

    The workspace branch is unchanged. The persistent high-resolution stream is
    NCHW, and local read/write is executed with depthwise conv kernels.
    """

    block_cls = LocalTHCNCHWSharedAdaLNJiTBlock
    final_cls = FinalLayerSharedAdaLN
    x_embedder_nchw_cls = BottleneckPatchEmbedNCHW

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.x_embedder = self.x_embedder_nchw_cls(
            self.input_size,
            self.patch_size,
            self.in_channels,
            128 if self.hidden_size <= 1024 else 256,
            self.hidden_size,
        )
        self.shared_adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(self.hidden_size, 6 * self.hidden_size, bias=True))
        self.shared_final_modulation = nn.Sequential(nn.SiLU(), nn.Linear(self.hidden_size, 2 * self.hidden_size, bias=True))
        self.initialize_weights()
        self._init_shared_adaln()

    def _init_shared_adaln(self) -> None:
        for module in (self.shared_adaLN_modulation[-1], self.shared_final_modulation[-1]):
            nn.init.constant_(module.weight, 0)
            nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        c = self.t_embedder(t) + self.y_embedder(y)
        t6 = self.shared_adaLN_modulation(c)
        t2 = self.shared_final_modulation(c)
        x_hi = self.x_embedder(x)
        for block in self.blocks:
            x_hi = block(x_hi, t6, self.workspace_rope)
        x_out = self.final_layer(x_hi.flatten(2).transpose(1, 2), t2)
        return self.unpatchify(x_out)


class LocalTHCNCHWFixedPoolSharedAdaLNJiT(LocalTHCNCHWSharedAdaLNJiT):
    """Fixed-pool speed reference for LocalTHC."""

    block_cls = LocalTHCNCHWFixedPoolSharedAdaLNJiTBlock


class LocalTHCNCHWDirectFixedPoolSharedAdaLNJiT(LocalTHCNCHWFixedPoolSharedAdaLNJiT):
    """Direct patch embed + fixed-pool read/write speed-oriented LocalTHC."""

    x_embedder_nchw_cls = DirectPatchEmbedNCHW


def LocalTHC_JiT_B_4(**kwargs) -> LocalTHCJiT:
    kwargs.setdefault("pool_groups", 768)
    return LocalTHCJiT(patch_size=4, hidden_size=768, depth=12, num_heads=12, bottleneck_dim=128, workspace_grid=16, **kwargs)


def LocalTHC_JiT_B_8(**kwargs) -> LocalTHCJiT:
    kwargs.setdefault("pool_groups", 768)
    return LocalTHCJiT(patch_size=8, hidden_size=768, depth=12, num_heads=12, bottleneck_dim=128, workspace_grid=16, **kwargs)


def LocalTHC_SharedAdaLN_JiT_B_4(**kwargs) -> LocalTHCSharedAdaLNJiT:
    kwargs.setdefault("pool_groups", 768)
    return LocalTHCSharedAdaLNJiT(patch_size=4, hidden_size=768, depth=12, num_heads=12, bottleneck_dim=128, workspace_grid=16, **kwargs)


def LocalTHC_Direct_SharedAdaLN_JiT_B_4(**kwargs) -> LocalTHCDirectSharedAdaLNJiT:
    kwargs.setdefault("pool_groups", 768)
    return LocalTHCDirectSharedAdaLNJiT(patch_size=4, hidden_size=768, depth=12, num_heads=12, bottleneck_dim=128, workspace_grid=16, **kwargs)


def LocalTHC_NCHW_SharedAdaLN_JiT_B_4(**kwargs) -> LocalTHCNCHWSharedAdaLNJiT:
    kwargs.setdefault("pool_groups", 768)
    return LocalTHCNCHWSharedAdaLNJiT(patch_size=4, hidden_size=768, depth=12, num_heads=12, bottleneck_dim=128, workspace_grid=16, **kwargs)


def LocalTHC_NCHW_FixedPool_SharedAdaLN_JiT_B_4(**kwargs) -> LocalTHCNCHWFixedPoolSharedAdaLNJiT:
    kwargs.setdefault("pool_groups", 768)
    return LocalTHCNCHWFixedPoolSharedAdaLNJiT(patch_size=4, hidden_size=768, depth=12, num_heads=12, bottleneck_dim=128, workspace_grid=16, **kwargs)


def LocalTHC_NCHW_Direct_FixedPool_SharedAdaLN_JiT_B_4(**kwargs) -> LocalTHCNCHWDirectFixedPoolSharedAdaLNJiT:
    kwargs.setdefault("pool_groups", 768)
    return LocalTHCNCHWDirectFixedPoolSharedAdaLNJiT(patch_size=4, hidden_size=768, depth=12, num_heads=12, bottleneck_dim=128, workspace_grid=16, **kwargs)


def LocalTHC_Triton_SharedAdaLN_JiT_B_4(**kwargs) -> LocalTHCTritonSharedAdaLNJiT:
    kwargs.setdefault("pool_groups", 768)
    return LocalTHCTritonSharedAdaLNJiT(patch_size=4, hidden_size=768, depth=12, num_heads=12, bottleneck_dim=128, workspace_grid=16, **kwargs)


def LocalTHC_Direct_Triton_SharedAdaLN_JiT_B_4(**kwargs) -> LocalTHCDirectTritonSharedAdaLNJiT:
    kwargs.setdefault("pool_groups", 768)
    return LocalTHCDirectTritonSharedAdaLNJiT(patch_size=4, hidden_size=768, depth=12, num_heads=12, bottleneck_dim=128, workspace_grid=16, **kwargs)


def LocalTHC_Direct_Triton_ReadAdd_SharedAdaLN_JiT_B_4(**kwargs) -> LocalTHCDirectTritonReadAddSharedAdaLNJiT:
    kwargs.setdefault("pool_groups", 768)
    return LocalTHCDirectTritonReadAddSharedAdaLNJiT(patch_size=4, hidden_size=768, depth=12, num_heads=12, bottleneck_dim=128, workspace_grid=16, **kwargs)


def LocalTHC_Direct_Triton_Frozen_SharedAdaLN_JiT_B_4(**kwargs) -> LocalTHCDirectTritonFrozenSharedAdaLNJiT:
    kwargs.setdefault("pool_groups", 768)
    return LocalTHCDirectTritonFrozenSharedAdaLNJiT(patch_size=4, hidden_size=768, depth=12, num_heads=12, bottleneck_dim=128, workspace_grid=16, **kwargs)


def LocalTHC_Direct_Triton_Traceable_SharedAdaLN_JiT_B_4(**kwargs) -> LocalTHCDirectTritonTraceableSharedAdaLNJiT:
    kwargs.setdefault("pool_groups", 768)
    return LocalTHCDirectTritonTraceableSharedAdaLNJiT(patch_size=4, hidden_size=768, depth=12, num_heads=12, bottleneck_dim=128, workspace_grid=16, **kwargs)


def LocalTHC_CompilerFriendly_SharedAdaLN_JiT_B_4(**kwargs) -> LocalTHCCompilerFriendlySharedAdaLNJiT:
    kwargs.setdefault("pool_groups", 768)
    return LocalTHCCompilerFriendlySharedAdaLNJiT(patch_size=4, hidden_size=768, depth=12, num_heads=12, bottleneck_dim=128, workspace_grid=16, **kwargs)


def LocalTHC_Direct_CompilerFriendly_SharedAdaLN_JiT_B_4(**kwargs) -> LocalTHCDirectCompilerFriendlySharedAdaLNJiT:
    kwargs.setdefault("pool_groups", 768)
    return LocalTHCDirectCompilerFriendlySharedAdaLNJiT(patch_size=4, hidden_size=768, depth=12, num_heads=12, bottleneck_dim=128, workspace_grid=16, **kwargs)


def LocalTHC_SharedAdaLN_JiT_B_8(**kwargs) -> LocalTHCSharedAdaLNJiT:
    kwargs.setdefault("pool_groups", 768)
    return LocalTHCSharedAdaLNJiT(patch_size=8, hidden_size=768, depth=12, num_heads=12, bottleneck_dim=128, workspace_grid=16, **kwargs)


def LocalTHC_InContext_SharedAdaLN_JiT_B_8(**kwargs) -> LocalTHCInContextSharedAdaLNJiT:
    kwargs.setdefault("pool_groups", 768)
    return LocalTHCInContextSharedAdaLNJiT(
        patch_size=8,
        hidden_size=768,
        depth=12,
        num_heads=12,
        bottleneck_dim=128,
        workspace_grid=16,
        num_class_tokens=32,
        class_token_start=4,
        **kwargs,
    )


def LocalTHC_InContext_CompilerFriendly_SharedAdaLN_JiT_B_8(**kwargs) -> LocalTHCInContextCompilerFriendlySharedAdaLNJiT:
    kwargs.setdefault("pool_groups", 768)
    return LocalTHCInContextCompilerFriendlySharedAdaLNJiT(
        patch_size=8,
        hidden_size=768,
        depth=12,
        num_heads=12,
        bottleneck_dim=128,
        workspace_grid=16,
        num_class_tokens=32,
        class_token_start=4,
        **kwargs,
    )


def SharedWrite_LocalTHC_SharedAdaLN_JiT_B_4(**kwargs) -> SharedWriteLocalTHCSharedAdaLNJiT:
    kwargs.setdefault("pool_groups", 768)
    return SharedWriteLocalTHCSharedAdaLNJiT(patch_size=4, hidden_size=768, depth=12, num_heads=12, bottleneck_dim=128, workspace_grid=16, **kwargs)


def SharedWrite_FusedFinal_LocalTHC_SharedAdaLN_JiT_B_4(**kwargs) -> SharedWriteFusedFinalLocalTHCSharedAdaLNJiT:
    kwargs.setdefault("pool_groups", 768)
    return SharedWriteFusedFinalLocalTHCSharedAdaLNJiT(patch_size=4, hidden_size=768, depth=12, num_heads=12, bottleneck_dim=128, workspace_grid=16, **kwargs)


def SharedWrite_FusedFinal12_LocalTHC_SharedAdaLN_JiT_B_4(**kwargs) -> SharedWriteFusedFinal12LocalTHCSharedAdaLNJiT:
    kwargs.setdefault("pool_groups", 768)
    return SharedWriteFusedFinal12LocalTHCSharedAdaLNJiT(patch_size=4, hidden_size=768, depth=12, num_heads=12, bottleneck_dim=128, workspace_grid=16, **kwargs)


def SharedRead_LocalTHC_SharedAdaLN_JiT_B_4(**kwargs) -> SharedReadLocalTHCSharedAdaLNJiT:
    return SharedWrite_LocalTHC_SharedAdaLN_JiT_B_4(**kwargs)


def SharedRead_FusedFinal_LocalTHC_SharedAdaLN_JiT_B_4(**kwargs) -> SharedReadFusedFinalLocalTHCSharedAdaLNJiT:
    return SharedWrite_FusedFinal_LocalTHC_SharedAdaLN_JiT_B_4(**kwargs)


def SharedRead_FusedFinal12_LocalTHC_SharedAdaLN_JiT_B_4(**kwargs) -> SharedReadFusedFinal12LocalTHCSharedAdaLNJiT:
    return SharedWrite_FusedFinal12_LocalTHC_SharedAdaLN_JiT_B_4(**kwargs)


def TriangularLazy_FusedFinal12_LocalTHC_DirectTriton_SharedAdaLN_JiT_B_4(
    **kwargs,
) -> TriangularLazyFusedFinal12LocalTHCDirectTritonSharedAdaLNJiT:
    kwargs.setdefault("pool_groups", 768)
    return TriangularLazyFusedFinal12LocalTHCDirectTritonSharedAdaLNJiT(
        patch_size=4,
        hidden_size=768,
        depth=12,
        num_heads=12,
        bottleneck_dim=128,
        workspace_grid=16,
        **kwargs,
    )


def TriangularLazy_FusedFinal12_LocalTHC_TritonTraceable_SharedAdaLN_JiT_B_4(
    **kwargs,
) -> TriangularLazyFusedFinal12LocalTHCTritonTraceableSharedAdaLNJiT:
    kwargs.setdefault("pool_groups", 768)
    return TriangularLazyFusedFinal12LocalTHCTritonTraceableSharedAdaLNJiT(
        patch_size=4,
        hidden_size=768,
        depth=12,
        num_heads=12,
        bottleneck_dim=128,
        workspace_grid=16,
        **kwargs,
    )


def SharedWrite_LocalTHC_SharedAdaLN_JiT_B_8(**kwargs) -> SharedWriteLocalTHCSharedAdaLNJiT:
    kwargs.setdefault("pool_groups", 768)
    return SharedWriteLocalTHCSharedAdaLNJiT(patch_size=8, hidden_size=768, depth=12, num_heads=12, bottleneck_dim=128, workspace_grid=16, **kwargs)


def SharedRead_LocalTHC_SharedAdaLN_JiT_B_8(**kwargs) -> SharedReadLocalTHCSharedAdaLNJiT:
    return SharedWrite_LocalTHC_SharedAdaLN_JiT_B_8(**kwargs)


def LocalTHC_JiT_L_4(**kwargs) -> LocalTHCJiT:
    kwargs.setdefault("pool_groups", 1024)
    return LocalTHCJiT(patch_size=4, hidden_size=1024, depth=24, num_heads=16, bottleneck_dim=128, workspace_grid=16, **kwargs)


def LocalTHC_JiT_L_8(**kwargs) -> LocalTHCJiT:
    kwargs.setdefault("pool_groups", 1024)
    return LocalTHCJiT(patch_size=8, hidden_size=1024, depth=24, num_heads=16, bottleneck_dim=128, workspace_grid=16, **kwargs)


def LocalTHC_SharedAdaLN_JiT_L_4(**kwargs) -> LocalTHCSharedAdaLNJiT:
    kwargs.setdefault("pool_groups", 1024)
    return LocalTHCSharedAdaLNJiT(patch_size=4, hidden_size=1024, depth=24, num_heads=16, bottleneck_dim=128, workspace_grid=16, **kwargs)


def LocalTHC_SharedAdaLN_JiT_L_8(**kwargs) -> LocalTHCSharedAdaLNJiT:
    kwargs.setdefault("pool_groups", 1024)
    return LocalTHCSharedAdaLNJiT(patch_size=8, hidden_size=1024, depth=24, num_heads=16, bottleneck_dim=128, workspace_grid=16, **kwargs)


def LocalTHC_JiT_H_4(**kwargs) -> LocalTHCJiT:
    kwargs.setdefault("pool_groups", 1280)
    return LocalTHCJiT(patch_size=4, hidden_size=1280, depth=32, num_heads=16, bottleneck_dim=256, workspace_grid=16, **kwargs)


def LocalTHC_JiT_H_8(**kwargs) -> LocalTHCJiT:
    kwargs.setdefault("pool_groups", 1280)
    return LocalTHCJiT(patch_size=8, hidden_size=1280, depth=32, num_heads=16, bottleneck_dim=256, workspace_grid=16, **kwargs)


def LocalTHC_SharedAdaLN_JiT_H_4(**kwargs) -> LocalTHCSharedAdaLNJiT:
    kwargs.setdefault("pool_groups", 1280)
    return LocalTHCSharedAdaLNJiT(patch_size=4, hidden_size=1280, depth=32, num_heads=16, bottleneck_dim=256, workspace_grid=16, **kwargs)


def LocalTHC_SharedAdaLN_JiT_H_8(**kwargs) -> LocalTHCSharedAdaLNJiT:
    kwargs.setdefault("pool_groups", 1280)
    return LocalTHCSharedAdaLNJiT(patch_size=8, hidden_size=1280, depth=32, num_heads=16, bottleneck_dim=256, workspace_grid=16, **kwargs)


LocalTHC_JiT_models = {
    "LocalTHC-JiT-B/4": LocalTHC_JiT_B_4,
    "LocalTHC-JiT-B/8": LocalTHC_JiT_B_8,
    "LocalTHC-SharedAdaLN-JiT-B/4": LocalTHC_SharedAdaLN_JiT_B_4,
    "LocalTHC-Direct-SharedAdaLN-JiT-B/4": LocalTHC_Direct_SharedAdaLN_JiT_B_4,
    "LocalTHC-NCHW-SharedAdaLN-JiT-B/4": LocalTHC_NCHW_SharedAdaLN_JiT_B_4,
    "LocalTHC-NCHW-FixedPool-SharedAdaLN-JiT-B/4": LocalTHC_NCHW_FixedPool_SharedAdaLN_JiT_B_4,
    "LocalTHC-NCHW-Direct-FixedPool-SharedAdaLN-JiT-B/4": LocalTHC_NCHW_Direct_FixedPool_SharedAdaLN_JiT_B_4,
    "LocalTHC-Triton-SharedAdaLN-JiT-B/4": LocalTHC_Triton_SharedAdaLN_JiT_B_4,
    "LocalTHC-Direct-Triton-SharedAdaLN-JiT-B/4": LocalTHC_Direct_Triton_SharedAdaLN_JiT_B_4,
    "LocalTHC-Direct-Triton-ReadAdd-SharedAdaLN-JiT-B/4": LocalTHC_Direct_Triton_ReadAdd_SharedAdaLN_JiT_B_4,
    "LocalTHC-Direct-Triton-Frozen-SharedAdaLN-JiT-B/4": LocalTHC_Direct_Triton_Frozen_SharedAdaLN_JiT_B_4,
    "LocalTHC-Direct-Triton-Traceable-SharedAdaLN-JiT-B/4": LocalTHC_Direct_Triton_Traceable_SharedAdaLN_JiT_B_4,
    "LocalTHC-CompilerFriendly-SharedAdaLN-JiT-B/4": LocalTHC_CompilerFriendly_SharedAdaLN_JiT_B_4,
    "LocalTHC-Direct-CompilerFriendly-SharedAdaLN-JiT-B/4": LocalTHC_Direct_CompilerFriendly_SharedAdaLN_JiT_B_4,
    "LocalTHC-SharedAdaLN-JiT-B/8": LocalTHC_SharedAdaLN_JiT_B_8,
    "LocalTHC-InContext-SharedAdaLN-JiT-B/8": LocalTHC_InContext_SharedAdaLN_JiT_B_8,
    "LocalTHC-InContext-CompilerFriendly-SharedAdaLN-JiT-B/8": LocalTHC_InContext_CompilerFriendly_SharedAdaLN_JiT_B_8,
    "SharedRead-LocalTHC-SharedAdaLN-JiT-B/4": SharedRead_LocalTHC_SharedAdaLN_JiT_B_4,
    "SharedRead-FusedFinal-LocalTHC-SharedAdaLN-JiT-B/4": SharedRead_FusedFinal_LocalTHC_SharedAdaLN_JiT_B_4,
    "SharedRead-FusedFinal12-LocalTHC-SharedAdaLN-JiT-B/4": SharedRead_FusedFinal12_LocalTHC_SharedAdaLN_JiT_B_4,
    "SharedWrite-LocalTHC-SharedAdaLN-JiT-B/4": SharedWrite_LocalTHC_SharedAdaLN_JiT_B_4,
    "SharedWrite-FusedFinal-LocalTHC-SharedAdaLN-JiT-B/4": SharedWrite_FusedFinal_LocalTHC_SharedAdaLN_JiT_B_4,
    "SharedWrite-FusedFinal12-LocalTHC-SharedAdaLN-JiT-B/4": SharedWrite_FusedFinal12_LocalTHC_SharedAdaLN_JiT_B_4,
    "TriangularLazy-FusedFinal12-LocalTHC-DirectTriton-SharedAdaLN-JiT-B/4": TriangularLazy_FusedFinal12_LocalTHC_DirectTriton_SharedAdaLN_JiT_B_4,
    "TriangularLazy-FusedFinal12-LocalTHC-TritonTraceable-SharedAdaLN-JiT-B/4": TriangularLazy_FusedFinal12_LocalTHC_TritonTraceable_SharedAdaLN_JiT_B_4,
    "SharedRead-LocalTHC-SharedAdaLN-JiT-B/8": SharedRead_LocalTHC_SharedAdaLN_JiT_B_8,
    "SharedWrite-LocalTHC-SharedAdaLN-JiT-B/8": SharedWrite_LocalTHC_SharedAdaLN_JiT_B_8,
    "LocalTHC-JiT-L/4": LocalTHC_JiT_L_4,
    "LocalTHC-JiT-L/8": LocalTHC_JiT_L_8,
    "LocalTHC-SharedAdaLN-JiT-L/4": LocalTHC_SharedAdaLN_JiT_L_4,
    "LocalTHC-SharedAdaLN-JiT-L/8": LocalTHC_SharedAdaLN_JiT_L_8,
    "LocalTHC-JiT-H/4": LocalTHC_JiT_H_4,
    "LocalTHC-JiT-H/8": LocalTHC_JiT_H_8,
    "LocalTHC-SharedAdaLN-JiT-H/4": LocalTHC_SharedAdaLN_JiT_H_4,
    "LocalTHC-SharedAdaLN-JiT-H/8": LocalTHC_SharedAdaLN_JiT_H_8,
}
