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

    # Dont use this ones
    experiment_name: str = "test"
    job_name: str = "test"
    compute_model_distance: bool = False
    compute_base_model_loss: bool = True
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


def _test_eqvivalence_impl(model: str, dataset: str, opt: str, seeds = list[int], overhoot_factors = list[int]):
    if opt == "sgd":
        optimizers = ["sgd_momentum", "sgd_overshoot"]
    elif opt == "adam":
        optimizers = ["adamW", "adamW_overshoot_replication"]

    for seed in seeds:
        for overshoot in overhoot_factors:
            args = Args(seed=seed, model=model, dataset=dataset, overshoot_factor=overshoot)
            args.config_override = ["precision=high", "max_steps=200"]
            print(f"\n=== Testing eqvivalence of:")
            
            args.two_models = True
            args.opt_name = optimizers[0]
            print("Version 1: ", args)
            with suppress_print():
                r1 = main(args)
                
            args.two_models = False
            args.opt_name = optimizers[1]
            print("Version 2: ", args)
            with suppress_print():
                r2 = main(args)
                
            assert round(r1, 10) == round(r2, 10)
        
        
def test_eqvivalence_mlp_mnist():
    _test_eqvivalence_impl(model="2c2d", dataset="fmnist", opt="adam", seeds=[10], overhoot_factors=[0.9, 4])
    _test_eqvivalence_impl(model="mlp", dataset="mnist", opt="sgd", seeds=[20], overhoot_factors=[0.9, 4])
    _test_eqvivalence_impl(model="mlp", dataset="mnist", opt="adam", seeds=[30], overhoot_factors=[0.9, 4])
    
