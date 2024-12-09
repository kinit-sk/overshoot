import torch

from optimizers.sgd_overshoot import SGDO
from optimizers.backups2.sgdo_adaptive import SGDO as SGDO_adaptive
from optimizers.backups2.adamw_overshoot_replication import AdamW as OvershootAdamW_replication
from optimizers.backups2.adamw_overshoot_full_approximation import AdamW as OvershootAdamW_full_approximation
from optimizers.backups2.adamw_overshoot_denom_approximation import AdamW as OvershootAdamW_denom_approximation
from optimizers.adamw_overshoot_delayed import AdamO as OvershootAdamW_delayed
from optimizers.backups2.adamw_overshoot_adaptive import AdamW as OvershootAdamW_adaptive

optimizers_map = {
    "sgd": torch.optim.SGD,
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



def create_optimizer(opt_name, param_groups, overshoot_factor, lr, config, foreach=None):
    foreach=False
    if opt_name == "nadam":
        opt = optimizers_map[opt_name](
            param_groups,
            lr=lr,
            betas=(config.adam_beta1, config.adam_beta2),
            momentum_decay=1000000000000000000000000, # Turn of momentum decay
            weight_decay=config.weight_decay,
            decoupled_weight_decay=True,
            foreach=foreach,
        )
    elif opt_name == "adamW_overshoot_delayed":
        opt = optimizers_map[opt_name](
            param_groups,
            lr=lr,
            betas=(config.adam_beta1, config.adam_beta2),
            weight_decay=config.weight_decay,
            overshoot=overshoot_factor,
            overshoot_delay=config.overshoot_delay,
            foreach=foreach
        )
    elif opt_name == "adamW_overshoot_adaptive":
        opt = optimizers_map[opt_name](
            param_groups,
            lr=lr,
            betas=(config.adam_beta1, config.adam_beta2),
            weight_decay=config.weight_decay,
            cosine_target=config.target_cosine_similarity,
            foreach=foreach
        )
    elif opt_name.startswith("adamW_overshoot"):
        opt = optimizers_map[opt_name](
            param_groups,
            lr=lr,
            betas=(config.adam_beta1, config.adam_beta2),
            weight_decay=config.weight_decay,
            overshoot=overshoot_factor,
            foreach=foreach,
        )
    elif "adam" in opt_name:
        config.adam_beta1 *= "zero" not in opt_name
        opt = optimizers_map[opt_name](
            param_groups,
            lr=lr,
            betas=(config.adam_beta1, config.adam_beta2),
            weight_decay=config.weight_decay,
            foreach=foreach,
        )
    elif "sgd_adaptive" in opt_name:
        opt = optimizers_map[opt_name](
            param_groups,
            lr=lr,
            momentum=config.sgd_momentum,
            weight_decay=config.weight_decay,
            cosine_target=config.target_cosine_similarity,
            foreach=foreach,
        )
    elif "sgd_overshoot" in opt_name:
        opt = optimizers_map[opt_name](
            param_groups,
            lr=lr,
            momentum=config.sgd_momentum,
            weight_decay=config.weight_decay,
            overshoot=overshoot_factor,
            foreach=foreach,
        )
    elif "sgd" in opt_name:
        opt = optimizers_map[opt_name](
            param_groups,
            lr=lr,
            momentum=0 if opt_name == "sgd" else config.sgd_momentum,
            weight_decay=config.weight_decay,
            nesterov="nesterov" in opt_name,
            foreach=foreach,
        )
    else:
        raise Exception(f"Optimizer {opt_name} not recognized.")
    return opt