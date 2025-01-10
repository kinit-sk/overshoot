
import argparse
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


task_2_title = {
    "mlp_housing": "MLP-CA",
    "vae_f-mnist": "VAE-FM",
    "vae_mnist": "VAE-M",
    "2c2d_f-mnist": "2c2d-FM",
    "3c3d_cifar10": "3c3d-C10",
    "resnet18_cifar100": "ResNet-C100",
}



def plot_data(data_90, data_95, task_index, title):

    for i, data in enumerate([data_90, data_95]):
        pp = axs[i, task_index]
        x_coords = [x[0] for x in data]
        y_coords = [x[1] for x in data]
        values = [x[2] for x in data]

        
        scatter = pp.scatter(x_coords, y_coords, c=values, cmap="viridis", s=500)
        # scatter = pp.scatter(x_coords, y_coords, c=values, cmap="gray", s=240)

        cbar = fig.colorbar(scatter, ax=pp, orientation="horizontal", location='top', pad=0.13, format="{x:.2f}")

        if i == 0:
            pp.set_title(title, fontsize=28, pad=100)
        cbar.set_label("Training loss", fontsize=24)
        
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

        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="../lightning_logs/graph2")
    parser.add_argument("--out", type=str, default="distances.png")
    args = parser.parse_args()

    plt.rc('xtick', labelsize=14)  # X-axis tick labels font size
    plt.rc('ytick', labelsize=14)  # Y-axis tick labels font size
    fig, axs = plt.subplots(2, 6, figsize=(40, 14))
    fig.text(0.5, 0.01, 'Overshoot factor', ha='center', fontsize=40)
    fig.text(0.03, 0.5, 'Average weighted distance', va='center', rotation='vertical', fontsize=34)
    fig.text(0.01, 0.25, r"$\beta$1 = 0.95", va='center', rotation='vertical', fontsize=30)
    fig.text(0.01, 0.7, r"$\beta$1 = 0.9", va='center', rotation='vertical', fontsize=30)
    all_results = {}


    task_index = -1
    for task_name in [task_name for task_name in task_2_title if os.path.isdir(os.path.join(args.root, task_name))]:
        task_root = os.path.join(args.root, task_name)
        task_root_95 = os.path.join(args.root, f"{task_name}_95")
        print(f"Processing {task_name}")
        task_index += 1
        task_results, task_results_95 = [], []
        for run_name in os.listdir(task_root):
            run_root = os.path.join(task_root, run_name)
            if os.path.isdir(run_root):
                distance, loss = process_run(run_root)
                task_results.append((float(run_name.split('_')[-1]), distance, loss))
                
        if os.path.isdir(task_root_95):
            for run_name in os.listdir(task_root_95):
                run_root = os.path.join(task_root_95, run_name)
                if os.path.isdir(run_root):
                    distance, loss = process_run(run_root)
                    task_results_95.append((float(run_name.split('_')[-1]), distance, loss))


        # plot_data(task_results, axs[task_index % 2, task_index // 2], task_2_title[task_name])
        plot_data(task_results, task_results_95, task_index, task_2_title[task_name])


        print(f"Updated with {task_name}")
        plt.tight_layout(rect=[0.04, 0.07, 1, 1])  # Leave space for the global labels
        fig.subplots_adjust(hspace=0.08)
        plt.savefig(args.out)
            