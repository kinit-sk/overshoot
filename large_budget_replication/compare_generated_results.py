import os
import json

import numpy as np


task_name_map = {
    "mnist_vae": "mnist_vae",
    "fmnist_vae": "fmnist_vae",
    "fmnist_2c2d": "fmnist_2c2d",
    "cifar10_3c3d": "cifar_3c3d",
}


# These are from https://github.com/fsschneider/DeepOBS/tree/master/deepobs/tensorflow/testproblems
# According to paper (https://arxiv.org/pdf/2007.01547), weight-decay wasn't finetuned
task_weight_decay_map = {
    "mnist_vae": 0,
    "fmnist_vae": 0,
    "fmnist_2c2d": 0,
    "cifar10_3c3d": 0.002,
}


optimizer_name_map = {
    "MomentumOptimizer": "sgd",
    "NAGOptimizer": "nesterov",
    "AdamOptimizer": "adam",
    "NadamOptimizer": "nadam",
}



def load_paper_results():
    # Path to this repo: https://github.com/SirRob1997/Crowded-Valley---Results
    path_to_repo = "/home/kopi/kinit/Crowded-Valley---Results"
    budget = "large_budget"
    scheduler = "cosine"
    root = os.path.join(path_to_repo, "results_main", budget, scheduler)
    tasks = ["mnist_vae", "fmnist_vae", "fmnist_2c2d", "cifar10_3c3d"]
    optimizers_name_map = {
        "MomentumOptimizer": "sgd",
        "NAGOptimizer": "sgd_nesterov",
        "AdamOptimizer": "adam",
        "NadamOptimizer": "nadam",
    }


    results = {}
    for task in tasks:
        results[task] = {}
        for optimizer in optimizers_name_map.keys():
            task_optimizer_root = os.path.join(root, task, optimizer)
            losses = []
            for setting_name in os.listdir(task_optimizer_root):
                name = os.listdir(os.path.join(task_optimizer_root, setting_name))[0]
                with open(os.path.join(task_optimizer_root, setting_name, name), "r") as f:
                    losses.append(json.load(f)["test_losses"])

            best_run_index = np.array(losses).min(axis=1).argmin() 
            results[task][optimizers_name_map[optimizer]] = losses[best_run_index]
    return results


def load_our_results():
    # Generate using `genrate_our_results.sh`
    path_to_results = "lightning_logs/large_budget_replication"
    tasks = ["mnist_vae", "fmnist_vae", "fmnist_2c2d", "cifar10_3c3d"]
    optimizers_name_map = {
        "MomentumOptimizer": "sgd",
        "NAGOptimizer": "sgd_nesterov",
        "AdamOptimizer": "adam",
        "NadamOptimizer": "nadam",
    }

    jobs = [f"{t}_{o}" for t in tasks for o in optimizers_name_map.values()]
    import code; code.interact(local=locals())




if __name__ == "__main__":
    # original_results = load_paper_results()
    load_our_results()

    