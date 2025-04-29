import copy
import os
import sys
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
    compute_model_distance: bool = False
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


def _test_eqvivalence_impl(model: str, dataset: str, seeds = list[int]):
    # (opt_name, two_models, compute_base_model_loss_validation), overshoot
    eq_runs = [
        (("sgd_momentum", True, True), ("sgd_overshoot", False, True), 3),
        (("sgd_nesterov", False, True), ("sgd_overshoot", False, False), 0.9),
        (("adamW", True, True), ("adamW_overshoot_replication", False, True), 6),
    ]

    for run1, run2, overshoot in eq_runs:
        for seed in seeds:
            args = Args(seed=seed, model=model, dataset=dataset, overshoot_factor=overshoot)
            args.config_override = ["precision=high", "max_steps=100"]
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
        
        
def test_eqvivalence_mlp_mnist():
    _test_eqvivalence_impl(model="mlp", dataset="housing", seeds=[42])
    _test_eqvivalence_impl(model="2c2d", dataset="fmnist", seeds=[10])
    _test_eqvivalence_impl(model="mlp", dataset="mnist", seeds=[20])
    
