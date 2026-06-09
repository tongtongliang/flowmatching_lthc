"""Checkpoint loading helpers with read/write key compatibility."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from .models.local_thc import migrate_lthc_state_dict_keys


def migrate_checkpoint_state(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Return a state dict using README-consistent read/write key names."""
    return migrate_lthc_state_dict_keys(dict(state))


def load_model_state(model: torch.nn.Module, ckpt: Mapping, key: str = "ema") -> None:
    """Load ``model`` or ``ema`` weights while accepting old research keys.

    Old checkpoints used ``shared_write`` / ``blocks.*.read`` and parameter
    names ``write_logits`` / ``read_weight``. The public code uses
    ``shared_read`` / ``blocks.*.write`` and ``read_logits`` / ``write_weight``.
    """
    if key == "model":
        model.load_state_dict(migrate_checkpoint_state(ckpt["model"]))
        return

    if key != "ema":
        raise ValueError(f"unsupported checkpoint state key: {key}")
    if "ema" not in ckpt:
        raise KeyError("checkpoint does not contain EMA weights")

    ema = migrate_checkpoint_state(ckpt["ema"])
    state = model.state_dict()
    missing = []
    for name in state:
        if name in ema:
            state[name] = ema[name]
        else:
            missing.append(name)
    model.load_state_dict(state)

