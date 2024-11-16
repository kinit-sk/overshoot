
import argparse
import os

import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt





optimizers_names_mapping = {
    "sgd_baseline": "CM",
    "nesterov": "NAG",
    "sgd_overshoot_3": "SGD3",
    "sgd_overshoot_5": "SGD5",
    "sgd_overshoot_7": "SGD7",
    "sgd_overshoot_adaptive": "SGDA",
    "adam_baseline": "Adam",
    "nadam": "Nadam",
    "adam_overshoot_3": "Adam3",
    "adam_overshoot_5": "Adam5",
    "adam_overshoot_7": "Adam7",
    "adam_overshoot_adaptive": "AdamA",
}

task_name_mapping = {
    "mlp_housing": "MLP Housing (loss)",
    "mlp_mnist": "Mnist (acc)",
    "2c2d_fashion": "2c2c F-MNIST (acc)",
    "3c3d_cifar10": "3c3d CIFAR-10 (acc)",
    "vae_mnist": "VAE MNIST (loss)",
    "vae_f-mnist": "VAE F-MNIST (loss)",
}

rows_to_drop = [
    "Mnist (acc)"
]
columns_to_drop = [
    "SGDA",
    "AdamA",
]

def bold_min(row):
    # import code; code.interact(local=locals())
    if "acc" in row.name:
        fn_to_use = max
    elif "loss" in row.name:
        fn_to_use = min
    else:
        raise Exception(f"task name have to contain either acc or loss: {row.name}")
        
    sgd_min = fn_to_use([x[1][0] for x in row.items() if 'SGD' in x[0] or 'CM' in x[0]])
    adam_min = fn_to_use([x[1][0] for x in row.items() if 'Adam' in x[0]])
    return [f"\\textbf{{{val[0]:.2f} \u00B1{val[1]:.2f}}}" if (val[0] == sgd_min or val[0] == adam_min) else f"{val[0]:.2f} \u00B1{val[1]:.2f}" for _, val in row.items()]


def plot_data(data, pp, title):
    for label, loss in data.items():
        pp.plot(loss, label=label)
    pp.set_yscale("symlog")
    pp.set_title(title)
    pp.legend(loc="upper left")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    args = parser.parse_args()


    root = "../lightning_logs/table1"
    
    fig, axs = plt.subplots(2, len(task_name_mapping.keys()), figsize=(52, 20))
    all_results = {}
    for task_index, task_name in enumerate(os.listdir(root)):
        task_root = os.path.join(root, task_name)

        if task_name == "mlp_mnist":
            continue

        sgd_results = {}
        adam_results = {}
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
                    run_losses = np.zeros((n_seeds, (len(losses) // 6) - 200))
                run_losses[i] = losses[200:len(losses) // 6]

            if run_losses is None or "adaptive" in run_name:
                continue
            elif ("sgd" in run_name) or ("nesterov" in run_name):
                sgd_results[run_name] = run_losses.mean(0)
            else:
                adam_results[run_name] = run_losses.mean(0)
            
            
        if sgd_results:
            plot_data(sgd_results, axs[0, task_index], task_name)
        if adam_results:
            plot_data(adam_results, axs[1, task_index], task_name)

        print(f"Updated with {task_name}")
        fig.suptitle('SGD', fontsize=16)
        plt.tight_layout()
        plt.savefig('foo.png')
            



    # pd.read_csv()