
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
    "3c3d_cifar10": "3c3d-C10",
    "vae_mnist": "VAE-M",
    "vae_f-mnist": "VAE-FM",
    "resnet18_cifar100": "ResNet-C100",
}



def plot_data(data, pp, title):

    x_coords = [x[0] for x in data]
    y_coords = [x[1] for x in data]
    values = [x[2] for x in data]

     
    scatter = pp.scatter(x_coords, y_coords, c=values, cmap="viridis", s=500)
    # scatter = pp.scatter(x_coords, y_coords, c=values, cmap="gray", s=240)

    cbar = fig.colorbar(scatter, ax=pp, orientation="horizontal", label="Train loss", location='top', pad=0.11, format="{x:.2f}")
    # cbar = fig.colorbar(scatter, ax=pp, orientation="horizontal", label="Train loss", location='top', pad=0.12)

    pp.set_title(title, fontsize=28, pad=100)
    cbar.set_label("Train loss", fontsize=24)
    
    cbar.ax.tick_params(labelsize=18)
    pp.xaxis.set_tick_params(labelsize=18)
    pp.yaxis.set_tick_params(labelsize=18)
    
    cbar.ax.xaxis.set_label_position('bottom')
    cbar.ax.xaxis.set_ticks_position('bottom')



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
        
        
        loss = train_stats['base_loss_1'].to_numpy()
        # loss = train_stats['overshoot_loss_1'].to_numpy()
        # loss = np.mean(np.partition(loss, 100)[:100])
        # loss = np.log(loss).mean()
        # import code; code.interact(local=locals())
        loss = loss.mean()

        run_distances.append(np.mean(distances))
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
    fig, axs = plt.subplots(1, 6, figsize=(40, 8))
    fig.text(0.5, 0.01, 'Overshoot factor', ha='center', fontsize=40)
    fig.text(0.01, 0.5, 'Average weighted distance', va='center', rotation='vertical', fontsize=34)
    all_results = {}


    task_index = -1
    for task_name in [task_name for task_name in task_2_title if os.path.isdir(os.path.join(args.root, task_name))]:
        task_root = os.path.join(args.root, task_name)
        print(f"Processing {task_name}")
        task_index += 1
        task_results = []
        for run_name in os.listdir(task_root):
            # if '15' in run_name or '14' in run_name or '13' in run_name:
            #     continue
            run_root = os.path.join(task_root, run_name)
            if os.path.isdir(run_root):
                distance, loss = process_run(run_root)
                # task_results.append((int(run_name.split('_')[-1]), distance, loss))
                task_results.append((float(run_name.split('_')[-1]), distance, loss))


        # plot_data(task_results, axs[task_index % 2, task_index // 2], task_2_title[task_name])
        plot_data(task_results, axs[task_index], task_2_title[task_name])


        print(f"Updated with {task_name}")
        plt.tight_layout(rect=[0.02, 0.07, 1, 1])  # Leave space for the global labels
        plt.savefig(args.out)
            