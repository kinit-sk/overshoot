import torch

from optimizers.sgdo import SGDO
from optimizers.sgdo_adaptive import SGDO as SGDO_adaptive
from optimizers.adamw_overshoot_replication import AdamW as OvershootAdamW_replication
from optimizers.adamw_overshoot_full_approximation import AdamW as OvershootAdamW_full_approximation
from optimizers.adamw_overshoot_denom_approximation import AdamW as OvershootAdamW_denom_approximation
from optimizers.adamw_overshoot_delayed import AdamW as OvershootAdamW_delayed
from optimizers.adamw_overshoot_adaptive import AdamW as OvershootAdamW_adaptive

optimizers_map = {
    "sgd_momentum": torch.optim.SGD,
    "sgd_nesterov": torch.optim.SGD,
    "sgd_overshoot": SGDO,
    "sgd_adaptive": SGDO_adaptive,
    "adam": torch.optim.Adam,
    "adamW": torch.optim.AdamW,
    "adam_zero": torch.optim.Adam,
    "adamW_zero": torch.optim.AdamW,
    "nadam": torch.optim.NAdam,
    "adamW_overshoot_replication": OvershootAdamW_replication,
    "adamW_overshoot_full_approximation": OvershootAdamW_full_approximation,
    "adamW_overshoot_denom_approximation": OvershootAdamW_denom_approximation,
    "adamW_overshoot_delayed": OvershootAdamW_delayed,
    "adamW_overshoot_adaptive": OvershootAdamW_adaptive,
    "rmsprop": torch.optim.RMSprop,
}