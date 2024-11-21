
import argparse
import os

import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt




tasks_to_skip = [
    "mlp_housing_v2"
]

task_name_mapping = {
    "vae_f-mnist": "VAE-FM",
    "vae_mnist_replicate": "VAE-M",
    "2c2d_fashion_replicate": "2c2d-FM",
    "3c3d_cifar10_replicate": "3c3d-C10",
    "resnet18_cifar100_v2_better_aug_weight_decay_new_sgd": "ResNet-C100",
}

color_map = {
    "sgd_baseline": [1, 0, 0],
    "sgd_overshoot_3": [0.6, 0.4, 0],
    "sgd_overshoot_5": [0.6, 0, 0.4],
    "sgd_overshoot_7": [0.6, 0.2, 0.2],
    "adam_baseline": [0, 0, 1],
    "adam_overshoot_3": [0.4, 0, 0.6],
    "adam_overshoot_5": [0, 0.4, 0.6],
    "adam_overshoot_7": [0.2, 0.2, 0.6],
}


def plot_data(data, pp, title, use_legend=True):

    for label, color in color_map.items():
        line_style = 'dotted' if "baseline" in label else 'solid'
        pp.plot(data[label], label=label, color=color, linestyle=line_style)

    pp.set_yscale('log', base=10)
    pp.set_title(title)
    if use_legend:
        pp.legend(loc="upper left")
        # pp.legend(color_map.keys(), loc="upper left",)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    args = parser.parse_args()


    root = "../lightning_logs/table1"
    
    fig, axs = plt.subplots(2, 3, figsize=(20, 10))
    all_results = {}


    task_index = -1
    for task_name in os.listdir(root):
        task_root = os.path.join(root, task_name)
        if (not os.path.isdir(task_root)) or (task_name not in task_name_mapping):
            continue
        
        print(f"Processing {task_name}")
        task_index += 1

        if task_name == "mlp_mnist":
            continue

        task_results = {}
        for run_name in os.listdir(task_root):
            run_root = os.path.join(task_root, run_name)
            if not os.path.isdir(run_root):
                continue
            
            resutls = []
            run_losses = None
            n_seeds = len(os.listdir(run_root))
            for i, seed_id in enumerate(os.listdir(run_root)):
                finel_path = os.path.join(run_root, seed_id)
            
                stats_path = os.path.join(finel_path, "training_stats.csv")
                    
                if not os.path.exists(stats_path):
                    continue
                losses = pd.read_csv(stats_path)['base_loss_100']
                if run_losses is None:
                    run_losses = np.zeros((n_seeds, (len(losses) // 8)))
                if "housing" in task_name:
                    run_losses[i] = 100 * losses[:len(losses) // 8]
                else:
                    run_losses[i] = losses[:len(losses) // 8]

            if run_losses is None or "adaptive" in run_name or "nesterov" in run_name or "nadam" in run_name:
                continue
            else:
                task_results[run_name] = run_losses.mean(0)
            
        if task_results:
            plot_data(task_results, axs[task_index % 2, task_index // 2], task_name, task_index == 0)

        print(f"Updated with {task_name}")
        fig.suptitle('SGD', fontsize=16)
        plt.tight_layout()
        plt.savefig('foo_test.png')
            



    # pd.read_csv()