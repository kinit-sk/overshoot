import os
import sys
import pytest
from contextlib import contextmanager
from dataclasses import dataclass

from main import main

@dataclass
class Args:
    # Run specific
    two_models: bool = False
    opt_name: str = ""
    
    # Experiment specific
    seed: int = 0
    overshoot_factor: float = 0
    model: str = ""
    dataset: str = ""
    config_override: list[str] | None = None
    compute_base_model_loss_validation: bool = True

    # Dont use this ones
    experiment_name: str = "test"
    job_name: str = "test"
    compute_model_distance_f: int = 0
    compute_base_model_loss: bool = False
    compute_cosine: bool = False
    from_large_budget: bool = False


@contextmanager
def suppress_print():
    original_stdout = sys.stdout 
    sys.stdout = open(os.devnull, 'w')
    try:
        yield
    finally:
        sys.stdout.close()
        sys.stdout = original_stdout



# (opt_name, two_models, compute_base_model_loss_validation), overshoot
eq_optimizer_setups = [
    (("sgd_momentum", True, True), ("sgd_overshoot", False, True)),
    (("sgd_nesterov", False, True), ("sgd_overshoot", False, False)), # Requires to have overshoot == momentum
    (("adamW", True, True), ("adamW_overshoot_replication", False, True)),
]

@pytest.mark.parametrize("seed,overshoot,model,dataset,run1,run2", [
    (42, 3.0, "mlp", "housing", *eq_optimizer_setups[0]),
    (42, 0.9, "mlp", "housing", *eq_optimizer_setups[1]),
    (42, 6.0, "mlp", "fmnist", *eq_optimizer_setups[2]),
    (42, 0.9, "3c3d", "mnist",  *eq_optimizer_setups[1]),
])
def test_eqvivalence(model, dataset, seed, overshoot, run1, run2):
    args = Args(seed=seed, model=model, dataset=dataset, overshoot_factor=overshoot)
    args.config_override = ["precision=high", "max_steps=100", "weight_decay_adam=0", "use_lr_scheduler=False", "grad_clip=None"]
    print(f"\n=== Testing eqvivalence of:")
    
    args.opt_name = run1[0]
    args.two_models = run1[1]
    args.compute_base_model_loss_validation = run1[2]
    print("Version 1: ", args)
    with suppress_print():
        r1 = main(args)
        
    args.opt_name = run2[0]
    args.two_models = run2[1]
    args.compute_base_model_loss_validation = run2[2]
    print("Version 2: ", args)
    with suppress_print():
        r2 = main(args)
        
    assert round(r1, 10) == round(r2, 10)
       