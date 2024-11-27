
import argparse
import os

import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter1d
from matplotlib.ticker import FuncFormatter, MaxNLocator




task_2_title = {
    # "mlp_housing": "Housing_42",
    # "mlp_housing_seed_420": "Housing_420",
    # "mlp_housing_seed_421": "Housing_421",
    # "mlp_housing_all": "Housing_old",
    "2c2d_f-mnist": "2c2d",
    # "vae_mnist": "VAE-M",
    # "2c2d_fashion": "2c2d-FM",
    # "3c3d_cifar10": "3c3d-C10",
    # "resnet18_cifar100": "ResNet-C100",
}



def plot_data(data, pp, title):

    x_coords = [x[0] for x in data]
    y_coords = [x[1] for x in data]
    values = [x[2] for x in data]

    # Create a scatter plot
    scatter = pp.scatter(x_coords, y_coords, c=values, cmap="viridis", s=200)

    # Add text inside the dots
    # for item in data:
    #     pp.text(item[0], item[1], f"{100 * item[2]:.2f}", color="white", ha="center", va="center", fontsize=6)


    # Set labels and title
    pp.set_title(title, fontsize=20)
    cbar = fig.colorbar(scatter, ax=pp, orientation="vertical", label="Train loss")
    cbar.ax.tick_params(labelsize=8)  # Adjust color bar tick size

def process_run(run_root):
    # training_stats = os.path.join(run_root, "version_1", "training_stats.csv")
    # test_stats = os.path.join(run_root, "version_1", "test_stats.csv")

    run_distances, run_losses = [], []
    for seed_id in os.listdir(run_root):
        training_stats = os.path.join(run_root, seed_id, "training_stats.csv")
        test_stats = os.path.join(run_root, seed_id, "test_stats.csv")
    
        if not os.path.exists(training_stats):
            print(f"Warning, file {training_stats} does not exists")
            continue
            
        if not os.path.exists(test_stats):
            print(f"Warning, file {test_stats} does not exists")
            continue
        
        train_stats = pd.read_csv(training_stats, on_bad_lines='skip')
        
        distances = train_stats['model_distance'].to_numpy()
        distances = distances[distances != -1]
        
        loss = train_stats['base_loss_100'].to_numpy()
        # loss = np.mean(np.partition(loss, 100)[:100])
        loss = loss.mean()
        # loss = loss.min()

        run_distances.append(np.mean(distances[50:]))
        run_losses.append(loss)
        
    # return np.mean(distances[50:]), minn 
    # import code; code.interact(local=locals())
    return np.mean(run_distances), np.mean(run_losses)

    if run_losses:
        min_len = min([len(x) for x in run_losses])
        return np.array([loss[:min_len] for loss in run_losses]).mean(0)
    else:
        print(f"Warning, no losses for {run_root}")
        return None

        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="../lightning_logs/graph2")
    parser.add_argument("--out", type=str, default="distances.png")
    args = parser.parse_args()

    plt.rc('xtick', labelsize=14)  # X-axis tick labels font size
    plt.rc('ytick', labelsize=14)  # Y-axis tick labels font size
    fig, axs = plt.subplots(2, 3, figsize=(20, 10))
    fig.text(0.5, 0.01, 'Overshoot factros', ha='center', fontsize=22)
    fig.text(0.01, 0.5, 'Awd', va='center', rotation='vertical', fontsize=22)
    all_results = {}


    task_index = -1
    for task_name in os.listdir(args.root):
        task_root = os.path.join(args.root, task_name)
        if (not os.path.isdir(task_root)) or (task_name not in task_2_title):
            continue
        
        print(f"Processing {task_name}")
        task_index += 1
        task_results = []
        for run_name in os.listdir(task_root):
            if '15' in run_name or '14' in run_name or '13' in run_name:
                continue
            run_root = os.path.join(task_root, run_name)
            if os.path.isdir(run_root):
                distance, loss = process_run(run_root)
                # task_results.append((int(run_name.split('_')[-1]), distance, loss))
                task_results.append((float(run_name.split('_')[-1]), distance, loss))


        plot_data(task_results, axs[task_index % 2, task_index // 2], task_2_title[task_name])


        print(f"Updated with {task_name}")
        plt.tight_layout(rect=[0.02, 0.04, 1, 1])  # Leave space for the global labels
        plt.savefig(args.out)
            