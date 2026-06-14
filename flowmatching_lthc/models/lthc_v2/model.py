"""LocalTHC v2 dense DiT models.

This file keeps the flowmatching architecture dense: the high-resolution
residual stream is accumulated and passed to the normal patch prediction head.
The v2 local interface is deliberately simple and linear:

    z_l[b, p, c] = sum_s x0[b, p, s, c] * alpha_l[c, s]
                 + sum_{j < l} gamma[l, j, c] * dz_j[b, p, c]
    x_L[b, p, s, c] = x0[b, p, s, c] + sum_j beta_j[c, s] * dz_j[b, p, c]

where gamma[l, j, c] = sum_s alpha_l[c, s] * beta_j[c, s].
There is no softmax, no eta, and no shared read.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..local_thc import (
    BottleneckPatchEmbedNHWC,
    DirectPatchEmbedNHWC,
    FinalLayer,
    FinalLayerSharedAdaLN,
    LabelEmbedder,
    TimestepEmbedder,
    VisionRotaryEmbeddingFast,
    WorkspaceJiTBranch,
    WorkspaceSharedAdaLNBranch,
)
from ..local_thc_triton_kernels import all_base_b4_12_split_traceable as triton_all_base_b4_12_split_traceable
from ..local_thc_triton_kernels import final_accumulate_b4_12_traceable as triton_final_accumulate_b4_12_traceable
from ..local_thc_triton_kernels import (
    triangular_accumulate_b4_12_ptr_active_traceable_ops as triton_triangular_active_ops,
)


class LinearChannelwisePool(nn.Module):
    """Per-layer channelwise linear read/write weights for one 4x4 local cell."""

    def __init__(
        self,
        high_grid: int,
        workspace_grid: int,
        hidden_size: int,
        read_init: float | None = None,
        write_init: float = 1.0,
    ) -> None:
        super().__init__()
        if high_grid != 64 or workspace_grid != 16:
            raise ValueError("v2 fused B/4 path currently requires high_grid=64 and workspace_grid=16")
        self.high_grid = high_grid
        self.workspace_grid = workspace_grid
        self.hidden_size = hidden_size
        self.cell_size = high_grid // workspace_grid
        self.cell_tokens = self.cell_size * self.cell_size
        if self.cell_tokens != 16:
            raise ValueError(f"expected 16 local cell tokens, got {self.cell_tokens}")
        read_value = (1.0 / self.cell_tokens) if read_init is None else float(read_init)
        self.read_weight = nn.Parameter(torch.full((hidden_size, self.cell_tokens), read_value))
        self.write_weight = nn.Parameter(torch.full((hidden_size, self.cell_tokens), float(write_init)))

    def alpha(self, dtype: torch.dtype | None = None) -> torch.Tensor:
        return self.read_weight if dtype is None else self.read_weight.to(dtype=dtype)

    def beta(self, dtype: torch.dtype | None = None) -> torch.Tensor:
        return self.write_weight if dtype is None else self.write_weight.to(dtype=dtype)


class LocalTHCV2SharedAdaLNBlock(nn.Module):
    """v2 block: local linear pool parameters plus shared-AdaLN workspace branch."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        high_grid: int,
        workspace_grid: int = 16,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        attn_backend: str = "flash",
    ) -> None:
        super().__init__()
        self.pool = LinearChannelwisePool(high_grid, workspace_grid, hidden_size)
        self.branch = WorkspaceSharedAdaLNBranch(
            hidden_size,
            num_heads,
            mlp_ratio=mlp_ratio,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            attn_backend=attn_backend,
        )

    def branch_update(self, z: torch.Tensor, t6: torch.Tensor, rope: VisionRotaryEmbeddingFast) -> torch.Tensor:
        return self.branch(z, t6, rope)


class LocalTHCV2AdaLNBlock(nn.Module):
    """v2 block with standard per-block AdaLN modulation."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        high_grid: int,
        workspace_grid: int = 16,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        attn_backend: str = "flash",
    ) -> None:
        super().__init__()
        self.pool = LinearChannelwisePool(high_grid, workspace_grid, hidden_size)
        self.branch = WorkspaceJiTBranch(
            hidden_size,
            num_heads,
            mlp_ratio=mlp_ratio,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            attn_backend=attn_backend,
        )

    def branch_update(self, z: torch.Tensor, c: torch.Tensor, rope: VisionRotaryEmbeddingFast) -> torch.Tensor:
        return self.branch(z, c, rope)


class _LocalTHCV2Base(nn.Module):
    """Common dense fused schedule for LocalTHC v2 B/4."""

    block_cls: type[nn.Module]
    final_cls: type[nn.Module]
    x_embedder_cls: type[nn.Module] = BottleneckPatchEmbedNHWC

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
        if input_size != 256 or patch_size != 4 or workspace_grid != 16 or depth != 12:
            raise ValueError("LocalTHC v2 fused kernel currently supports only B/4: input=256, patch=4, grid=16, depth=12")
        if pool_groups not in (None, hidden_size):
            raise ValueError("LocalTHC v2 uses per-channel pools; pool_groups must be None or hidden_size")
        self.input_size = input_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.workspace_grid = workspace_grid
        self.high_grid = input_size // patch_size
        self.num_high_tokens = self.high_grid * self.high_grid
        self.num_workspace_tokens = workspace_grid * workspace_grid
        self.pool_groups = hidden_size

        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size)
        self.x_embedder = self.x_embedder_cls(input_size, patch_size, in_channels, bottleneck_dim, hidden_size)
        half_head_dim = hidden_size // num_heads // 2
        self.workspace_rope = VisionRotaryEmbeddingFast(half_head_dim, workspace_grid, num_prefix_tokens=0)
        self.blocks = nn.ModuleList(
            [
                self.block_cls(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    high_grid=self.high_grid,
                    workspace_grid=workspace_grid,
                    mlp_ratio=mlp_ratio,
                    attn_drop=attn_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0,
                    proj_drop=proj_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0,
                    attn_backend=attn_backend,
                )
                for i in range(depth)
            ]
        )
        self.final_layer = self.final_cls(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

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
        self._init_adaln()
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def _init_adaln(self) -> None:
        for block in self.blocks:
            if hasattr(block.branch, "adaLN_modulation"):
                nn.init.constant_(block.branch.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.branch.adaLN_modulation[-1].bias, 0)
        if hasattr(self.final_layer, "adaLN_modulation"):
            nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        c = self.out_channels
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        if h * w != x.shape[1]:
            raise ValueError(f"token count {x.shape[1]} is not square")
        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = x.permute(0, 5, 1, 3, 2, 4)
        return x.reshape(x.shape[0], c, h * p, w * p)

    def _alpha_stack(self, dtype: torch.dtype) -> torch.Tensor:
        return torch.stack([block.pool.alpha(dtype=dtype) for block in self.blocks], dim=0).contiguous()

    def _beta_stack(self, dtype: torch.dtype) -> torch.Tensor:
        return torch.stack([block.pool.beta(dtype=dtype) for block in self.blocks], dim=0).contiguous()

    def _final_accumulate(self, x0: torch.Tensor, dz_list: list[torch.Tensor]) -> torch.Tensor:
        if len(dz_list) != 12:
            raise ValueError("LocalTHC v2 fused final accumulation requires exactly 12 dz tensors")
        betas = [block.pool.beta(dtype=dz_list[0].dtype).contiguous() for block in self.blocks]
        return triton_final_accumulate_b4_12_traceable(x0, *dz_list, *betas)

    def _dense_output(self, x_hi: torch.Tensor, final_cond: torch.Tensor) -> torch.Tensor:
        b, hh, wh, c_dim = x_hi.shape
        x_out = self.final_layer(x_hi.reshape(b, hh * wh, c_dim), final_cond)
        return self.unpatchify(x_out)

    def _run_fused_schedule(self, x0: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        alpha_stack = self._alpha_stack(dtype=x0.dtype)
        beta_stack = self._beta_stack(dtype=x0.dtype)
        base_reads = triton_all_base_b4_12_split_traceable(x0, alpha_stack)
        gamma = (alpha_stack[:, None, :, :] * beta_stack[None, :, :, :]).sum(dim=-1)
        zero_dz = torch.zeros_like(base_reads[0])

        dz_list: list[torch.Tensor] = []
        for layer_idx, block in enumerate(self.blocks):
            z = base_reads[layer_idx]
            if dz_list:
                dz_args = tuple(dz_list[j] if j < layer_idx else zero_dz for j in range(12))
                z = triton_triangular_active_ops[layer_idx - 1](z, *dz_args, gamma[layer_idx].contiguous())
            dz = self._branch_update(block, z, cond)
            dz_list.append(dz)
        return self._final_accumulate(x0, dz_list)

    def _branch_update(self, block: nn.Module, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class LocalTHCV2FusedFinal12SharedAdaLNJiT(_LocalTHCV2Base):
    """Dense LocalTHC v2 with shared AdaLN modulation and fused B/4 schedule."""

    block_cls = LocalTHCV2SharedAdaLNBlock
    final_cls = FinalLayerSharedAdaLN

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.shared_adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(self.hidden_size, 6 * self.hidden_size, bias=True))
        self.shared_final_modulation = nn.Sequential(nn.SiLU(), nn.Linear(self.hidden_size, 2 * self.hidden_size, bias=True))
        for module in (self.shared_adaLN_modulation[-1], self.shared_final_modulation[-1]):
            nn.init.constant_(module.weight, 0)
            nn.init.constant_(module.bias, 0)

    def _branch_update(self, block: nn.Module, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return block.branch_update(z, cond, self.workspace_rope)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        c = self.t_embedder(t) + self.y_embedder(y)
        t6 = self.shared_adaLN_modulation(c)
        t2 = self.shared_final_modulation(c)
        x0 = self.x_embedder(x)
        x_hi = self._run_fused_schedule(x0, t6)
        return self._dense_output(x_hi, t2)


class LocalTHCV2FusedFinal12AdaLNJiT(_LocalTHCV2Base):
    """Dense LocalTHC v2 with standard per-block AdaLN modulation."""

    block_cls = LocalTHCV2AdaLNBlock
    final_cls = FinalLayer

    def _branch_update(self, block: nn.Module, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return block.branch_update(z, cond, self.workspace_rope)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        c = self.t_embedder(t) + self.y_embedder(y)
        x0 = self.x_embedder(x)
        x_hi = self._run_fused_schedule(x0, c)
        return self._dense_output(x_hi, c)


class LocalTHCV2DirectFusedFinal12SharedAdaLNJiT(LocalTHCV2FusedFinal12SharedAdaLNJiT):
    x_embedder_cls = DirectPatchEmbedNHWC


class LocalTHCV2DirectFusedFinal12AdaLNJiT(LocalTHCV2FusedFinal12AdaLNJiT):
    x_embedder_cls = DirectPatchEmbedNHWC


def LocalTHCV2_FusedFinal12_SharedAdaLN_JiT_B_4(**kwargs) -> LocalTHCV2FusedFinal12SharedAdaLNJiT:
    return LocalTHCV2FusedFinal12SharedAdaLNJiT(
        patch_size=4,
        hidden_size=768,
        depth=12,
        num_heads=12,
        bottleneck_dim=128,
        workspace_grid=16,
        **kwargs,
    )


def LocalTHCV2_FusedFinal12_AdaLN_JiT_B_4(**kwargs) -> LocalTHCV2FusedFinal12AdaLNJiT:
    return LocalTHCV2FusedFinal12AdaLNJiT(
        patch_size=4,
        hidden_size=768,
        depth=12,
        num_heads=12,
        bottleneck_dim=128,
        workspace_grid=16,
        **kwargs,
    )


def LocalTHCV2_Direct_FusedFinal12_SharedAdaLN_JiT_B_4(**kwargs) -> LocalTHCV2DirectFusedFinal12SharedAdaLNJiT:
    return LocalTHCV2DirectFusedFinal12SharedAdaLNJiT(
        patch_size=4,
        hidden_size=768,
        depth=12,
        num_heads=12,
        bottleneck_dim=128,
        workspace_grid=16,
        **kwargs,
    )


def LocalTHCV2_Direct_FusedFinal12_AdaLN_JiT_B_4(**kwargs) -> LocalTHCV2DirectFusedFinal12AdaLNJiT:
    return LocalTHCV2DirectFusedFinal12AdaLNJiT(
        patch_size=4,
        hidden_size=768,
        depth=12,
        num_heads=12,
        bottleneck_dim=128,
        workspace_grid=16,
        **kwargs,
    )


LocalTHCV2_JiT_models = {
    "LocalTHCv2-FusedFinal12-SharedAdaLN-JiT-B/4": LocalTHCV2_FusedFinal12_SharedAdaLN_JiT_B_4,
    "LocalTHCv2-FusedFinal12-AdaLN-JiT-B/4": LocalTHCV2_FusedFinal12_AdaLN_JiT_B_4,
    "LocalTHCv2-Direct-FusedFinal12-SharedAdaLN-JiT-B/4": LocalTHCV2_Direct_FusedFinal12_SharedAdaLN_JiT_B_4,
    "LocalTHCv2-Direct-FusedFinal12-AdaLN-JiT-B/4": LocalTHCV2_Direct_FusedFinal12_AdaLN_JiT_B_4,
}
