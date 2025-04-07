import argparse
import os
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_paper_results(best_setting: bool = False):
    if best_setting:
        budget = "large_budget"
        scheduler = "cosine"
    else:
        budget = "oneshot"
        scheduler = "none"

    root = os.path.join(args.paper_root, "results_main", budget, scheduler)

    results = {}
    for task in tasks:
        for optimizer in optimizers_name_map.keys():
            task_optimizer_root = os.path.join(root, task, optimizer)

            # Choose best run from large budget
            losses = []
            for setting_name in os.listdir(task_optimizer_root):
                name = os.listdir(os.path.join(task_optimizer_root, setting_name))[0]
                with open(os.path.join(task_optimizer_root, setting_name, name), "r") as f:
                    losses.append(json.load(f)["test_losses"])

            best_run_index = np.array(losses).min(axis=1).argmin() 
            key = f"{task}_{optimizers_name_map[optimizer]}"
            results[key] = losses[best_run_index][1:] # First datapoint is before first epoch
    return results


def load_our_results(prefix: str = ""):
    results = {}
    for job in [f"{prefix}{t}_{o}" for t in tasks for o in optimizers_name_map.values()]:
        path = os.path.join(args.experiments_root, job, "version_1", "test_stats.csv")
        if os.path.exists(path):
            results[job[len(prefix):]] = pd.read_csv(path)["loss"].tolist()
    return results


def vis():
    # assert our_results.keys() == paper_results.keys()
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    for key in our_paper_results.keys():
        plt.figure(figsize=(10, 6))
        plt.plot(paper_oneshot_results[key], label="Crowded Valley (one shot)", linestyle='dashed')
        plt.plot(paper_best_results[key], label="Crowded Valley (large budget)", linestyle='dashed')
        plt.plot(our_paper_results[key], label="Overshoot paper (one shot)")
        plt.plot(our_replication_results[key], label="Our Replication (large budget)")
        plt.yscale('log')
        plt.title(key)
        plt.legend()
        plt.savefig(f"{args.output}/{key}.png")
        plt.clf()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper_root", type=str, required=True, help="Path to this repo: https://github.com/SirRob1997/Crowded-Valley---Results")
    parser.add_argument("--experiments_root", type=str, required=True) # e.g., lightning_logs/large_budget_replication
    parser.add_argument("--output", type=str, default='figures')
    args = parser.parse_args()
    
    tasks = ["mnist_vae", "fmnist_vae", "fmnist_2c2d", "cifar10_3c3d"]
    optimizers_name_map = {
        "MomentumOptimizer": "sgd_momentum",
        "NAGOptimizer": "sgd_nesterov",
        "AdamOptimizer": "adamW",
        "NadamOptimizer": "nadam",
    }

    paper_best_results = load_paper_results(best_setting = True)
    paper_oneshot_results = load_paper_results(best_setting = False)
    our_replication_results = load_our_results("")
    our_paper_results = load_our_results("from_overshoot_paper_")
    vis()

    