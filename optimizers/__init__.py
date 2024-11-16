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



def create_optimizer(opt_name, param_groups, overshoot_factor, lr, config):
    if opt_name == "nadam":
        opt = optimizers_map[opt_name](
            param_groups,
            lr=lr,
            betas=(config.adam_beta1, config.adam_beta2),
            momentum_decay=1000000000000000000000000, # Turn of momentum decay
            foreach=False,
        )
    elif opt_name == "adamW_overshoot_delayed":
        opt = optimizers_map[opt_name](
            param_groups,
            lr=lr,
            betas=(config.adam_beta1, config.adam_beta2),
            weight_decay=config.weight_decay,
            overshoot=overshoot_factor,
            overshoot_delay=config.overshoot_delay,
            foreach=False,
        )
    elif opt_name == "adamW_overshoot_adaptive":
        opt = optimizers_map[opt_name](
            param_groups,
            lr=lr,
            betas=(config.adam_beta1, config.adam_beta2),
            weight_decay=config.weight_decay,
            cosine_target=config.target_cosine_similarity,
            foreach=False,
        )
    elif opt_name.startswith("adamW_overshoot"):
        opt = optimizers_map[opt_name](
            param_groups,
            lr=lr,
            betas=(config.adam_beta1, config.adam_beta2),
            weight_decay=config.weight_decay,
            overshoot=overshoot_factor,
            foreach=False,
        )
    elif "adam" in opt_name:
        config.adam_beta1 *= "zero" not in opt_name
        opt = optimizers_map[opt_name](
            param_groups,
            lr=lr,
            betas=(config.adam_beta1, config.adam_beta2),
            weight_decay=config.weight_decay,
        )
    elif "sgd_adaptive" in opt_name:
        opt = optimizers_map[opt_name](
            param_groups,
            lr=lr,
            momentum=config.sgd_momentum,
            cosine_target=config.target_cosine_similarity,
            foreach=False,
        )
    elif "sgd_overshoot" in opt_name:
        opt = optimizers_map[opt_name](
            param_groups,
            lr=lr,
            momentum=config.sgd_momentum,
            overshoot=overshoot_factor,
            foreach=False,
        )
    elif "sgd" in opt_name:
        opt = optimizers_map[opt_name](
            param_groups,
            lr=lr,
            momentum=0 if opt_name == "sgd" else config.sgd_momentum,
            nesterov="nesterov" in opt_name,
            foreach=False,
        )
    else:
        raise Exception(f"Optimizer {opt_name} not recognized.")
    return opt