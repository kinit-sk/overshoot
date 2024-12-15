
import argparse
import os

import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter1d
from matplotlib.ticker import FuncFormatter, MaxNLocator





task_2_title = {
    "mlp_housing": "MLP-CA",
    "2c2d_f-mnist": "2c2d-FM",
    "vae_f-mnist": "VAE-FM",
    "3c3d_cifar10": "3c3d-C10",
    "vae_mnist": "VAE-M",
    "resnet18_cifar100": "ResNet-C100",
    "gpt_hf_qqp": "GPT-2-GLUE",
}



task_2_range_train = {
    "mlp_housing": (0.11, 0.25),
    "vae_f-mnist": (22, 30),
    "vae_mnist": (26, 60),
    "2c2d_f-mnist": (0.0, 2),
    "3c3d_cifar10": (0.25, 4),
    "resnet18_cifar100": (0.0, 4),
    "gpt_hf_qqp": (0.1, 1),
}

task_2_range_test = {
    "mlp_housing": (0.25, 0.5),
    "vae_f-mnist": (22, 20),
    "vae_mnist": (26, 40),
    "2c2d_f-mnist": (0.22, 1),
    "3c3d_cifar10": (0.35, 3),
    "resnet18_cifar100": (1.7, 4),
    "gpt_hf_qqp": (0.28, 0.38),
}

task_2_smooth = {
    "mlp_housing": 20,
    "vae_f-mnist": 20,
    "vae_mnist": 22,
    "2c2d_f-mnist": 40,
    "3c3d_cifar10": 14,
    "resnet18_cifar100": 8,
    "gpt_hf_qqp": 10,
}

color_map = {
    "sgd_baseline": [1, 0, 0],
    "sgd_overshoot_3": [0.8, 0.4, 0],
    "sgd_overshoot_5": [0.6, 0.2, 0.2],
    "sgd_overshoot_7": [0.8, 0, 0.4],
    "adam_baseline": [0, 0, 1],
    "adam_overshoot_3": [0.4, 0, 0.8],
    "adam_overshoot_5": [0.2, 0.2, 0.6],
    "adam_overshoot_7": [0.0, 0.4, 0.8],
}

algorithm_2_legend = {
    "sgd_baseline": "SGD CM",
    "sgd_overshoot_3": r"SGDO($\gamma=3)$",
    "sgd_overshoot_5": r"SGDO($\gamma=5)$",
    "sgd_overshoot_7": r"SGDO($\gamma=7)$",
    "adam_baseline": "Adam",
    "adam_overshoot_3": r"AdamO($\gamma=3)$",
    "adam_overshoot_5": r"AdamO($\gamma=5)$",
    "adam_overshoot_7": r"AdamO($\gamma=7)$",
}


def plot_data(data, pp, task_name):

    pp.set_yscale('log', base=10)
    task_2_range = task_2_range_test if args.loss_type == "test" else task_2_range_train
    min_max = task_2_range[task_name]
    for label, color in color_map.items():
        if not label in data:
            continue
        line_style = 'dotted' if "baseline" in label else 'solid'
        pp.plot(data[label] - min_max[0], label=algorithm_2_legend[label], color=color, linestyle=line_style, linewidth=1.5)
        
    pp.set_ylim([0, min_max[1]])
    if args.loss_type == "test" and ("resnet" in task_name or "gpt" in task_name):
        pp.set_yticks(pp.get_yticks()[-5:-2], [round(min_max[0]+c, 2) for c in pp.get_yticks()[-5:-2]])
    else:
        pp.set_yticks(pp.get_yticks()[-4:-2], [round(min_max[0]+c, 2) for c in pp.get_yticks()[-4:-2]])

    if args.loss_type != "test":
        pp.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x * 10:.0f}"))
    pp.xaxis.set_major_locator(MaxNLocator(nbins=3)) 
    pp.set_title(task_2_title[task_name], fontsize=20)
    

def process_run(run_root, smooth_factor):
    run_losses = []
    for seed_id in os.listdir(run_root):
        finel_path = os.path.join(run_root, seed_id)
        if args.loss_type == "test":
            stats_path = os.path.join(finel_path, "test_stats.csv")
            key = "loss"
        else:
            stats_path = os.path.join(finel_path, "training_stats.csv")
            key = "overshoot_loss_1" if args.loss_type == "overshoot" else "base_loss_1"
             
        if not os.path.exists(stats_path):
            print(f"Warning, file {stats_path} does not exists")
            continue
        losses = pd.read_csv(stats_path, on_bad_lines='skip')[key].to_numpy()
         
        if args.loss_type != "test":
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
    parser.add_argument("--root", type=str, default="../lightning_logs/table1")
    parser.add_argument("--loss_type", type=str, default="base")
    args = parser.parse_args()

    assert args.loss_type in ["base", "overshoot", "test"]

    plt.rc('xtick', labelsize=14)  # X-axis tick labels font size
    plt.rc('ytick', labelsize=14)  # Y-axis tick labels font size
    fig, axs = plt.subplots(2, 4, figsize=(20, 10))
    if args.loss_type == "test":
        fig.text(0.5, 0.01, 'Epochs', ha='center', fontsize=22)
        fig.text(0.01, 0.5, 'Test loss', va='center', rotation='vertical', fontsize=22)
    else:
        fig.text(0.5, 0.01, 'Training steps', ha='center', fontsize=22)
        fig.text(0.01, 0.5, 'Training loss', va='center', rotation='vertical', fontsize=22)
    all_results = {}


    task_index = -1
    for task_name in [task_name for task_name in task_2_title if os.path.isdir(os.path.join(args.root, task_name))]:
        task_root = os.path.join(args.root, task_name)
        
        print(f"Processing {task_name}")
        task_index += 1
        task_index += task_index == 6
        task_results = {}
        for run_name in os.listdir(task_root):
            run_root = os.path.join(task_root, run_name)
            if os.path.isdir(run_root) and run_name in color_map.keys():
                task_results[run_name] = process_run(run_root, task_2_smooth[task_name])

        min_len = min([len(x) for x in task_results.values()])
        for k in task_results.keys():
            task_results[k] = task_results[k][:min_len // 1]

            
        if task_results:
            plot_data(task_results, axs[task_index % 2, task_index // 2], task_name)

        handles, labels = axs[0, 0].get_legend_handles_labels()
        legend = axs[0, -1].legend(handles, labels, loc="center",  frameon=False, fontsize=16) 
        for line in legend.get_lines():
            line.set_linewidth(3)
        
        
        axs[0, -1].axis('off') 
        print(f"Updated with {task_name}")
        plt.tight_layout(rect=[0.02, 0.04, 1, 1])  # Leave space for the global labels
        plt.savefig(f"graph1_{args.loss_type}_loss.png")
            