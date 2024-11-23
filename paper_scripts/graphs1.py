
import argparse
import os

import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter1d




# task_name_mapping = {
#     "mlp_housing_v4": "Housing",
#     "vae_f-mnist": "VAE-FM",
#     "vae_mnist_replicate": "VAE-M",
#     "2c2d_fashion_replicate": "2c2d-FM",
#     "3c3d_cifar10_replicate": "3c3d-C10",
#     "resnet18_cifar100_v2_better_aug_weight_decay_new_sgd": "ResNet-C100",
# }

task_2_title = {
    "mlp_housing": "Housing",
    "vae_f-mnist": "VAE-FM",
    "vae_mnist": "VAE-M",
    "2c2d_fashion": "2c2d-FM",
    "3c3d_cifar10": "3c3d-C10",
    "resnet18_cifar100": "ResNet-C100",
}


task_2_range = {
    "mlp_housing": (0.1, 1),
    "vae_f-mnist": (22, 30),
    "vae_mnist": (26, 60),
    "2c2d_fashion": (0.0, 2),
    "3c3d_cifar10": (0.25, 4),
    "resnet18_cifar100": (0.0, 4),
}

task_2_smooth = {
    "mlp_housing": 20,
    "vae_f-mnist": 20,
    "vae_mnist": 20,
    "vae_mnist": 20,
    "2c2d_fashion": 40,
    "3c3d_cifar10": 20,
    "resnet18_cifar100": 10,
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

algorithm_2_legend = {
    "sgd_baseline": "SGD CM",
    "sgd_overshoot_3": r"SGD $\gamma=3$",
    "sgd_overshoot_5": r"SGD $\gamma=5$",
    "sgd_overshoot_7": r"SGD $\gamma=7$",
    "adam_baseline": "Adam",
    "adam_overshoot_3": r"Adam $\gamma=3$",
    "adam_overshoot_5": r"Adam $\gamma=5$",
    "adam_overshoot_7": r"Adam $\gamma=7$",
}




def plot_data(data, pp, task_name, use_x_label=True, use_y_label=True, use_legend=True):
    
    pp.set_yscale('log', base=10)
    min_max = task_2_range[task_name]
    for label, color in color_map.items():
        line_style = 'dotted' if "baseline" in label else 'solid'
        pp.plot(data[label] - min_max[0], label=algorithm_2_legend[label], color=color, linestyle=line_style, linewidth=1.5)
        
    pp.set_ylim([0, min_max[1]])
    pp.set_yticks(pp.get_yticks()[-4:-2], [round(min_max[0]+c, 2) for c in pp.get_yticks()[-4:-2]])
    pp.set_title(task_2_title[task_name])
    if use_x_label:
        pp.set_xlabel('Batch')
    if use_y_label:
        pp.set_ylabel('Training Loss')
    if use_legend:
        pp.legend(loc="upper right")

def process_run(run_root, smooth_factor):
    run_losses = []
    for seed_id in os.listdir(run_root):
        finel_path = os.path.join(run_root, seed_id)
        stats_path = os.path.join(finel_path, "training_stats.csv")
        if not os.path.exists(stats_path):
            print(f"Warning, file {stats_path} does not exists")
            continue
        losses = pd.read_csv(stats_path, on_bad_lines='skip')['base_loss_1'].to_numpy()
         
        losses = losses[:(len(losses) // 10) * 10] # Make dividable by 20
        losses = losses.reshape(-1, 10).mean(axis=1)
        y_smooth = gaussian_filter1d(losses, smooth_factor)
        losses = y_smooth
        run_losses.append(losses)

    if run_losses:
        min_len = min([len(x) for x in run_losses])
        return np.array([loss[:min_len] for loss in run_losses]).mean(0)
    else:
        print(f"Warning, no losses for {run_root}")
        return None

        

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
        if (not os.path.isdir(task_root)) or (task_name not in task_2_title):
            continue
        
        print(f"Processing {task_name}")
        task_index += 1
        task_results = {}
        for run_name in os.listdir(task_root):
            run_root = os.path.join(task_root, run_name)
            if os.path.isdir(run_root) and run_name in color_map.keys():
                task_results[run_name] = process_run(run_root, task_2_smooth[task_name])

        min_len = min([len(x) for x in task_results.values()])
        for k in task_results.keys():
            task_results[k] = task_results[k][:min_len // 1]

            
        if task_results:
            use_x_label = task_index % 2 == 1
            use_y_label = task_index // 2 == 0
            use_legend = task_index == 4
            plot_data(task_results, axs[task_index % 2, task_index // 2], task_name, use_x_label, use_y_label, use_legend)

        print(f"Updated with {task_name}")
        plt.tight_layout()
        plt.savefig('out/foo_test.png')
            