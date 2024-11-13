
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


def plot_loss_graph(data_to_plot, data_type: str):
    plt.figure(figsize=(20, 16))
    for key, values in data_to_plot.items():
        loss_values = values.T
        loss_values = loss_values[100:, :] # Ignore start of the training to decrease variance
        mean_loss = np.mean(loss_values, axis=1)
        std_loss = 2 * np.std(loss_values, axis=1)

        # Plotting
        plt.plot(mean_loss, label=key)
        plt.fill_between(range(loss_values.shape[0]), mean_loss - std_loss, mean_loss + std_loss, alpha=0.2)
        
    plt.yscale('log')
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.title('Training Loss with Variance')
    plt.legend()
    plt.savefig(os.path.join(source_dir, f"{data_type}.png"))
    plt.clf()

def plot_data(data_to_plot, data_type: str):
    # Prepare data for plotting
    experiments, means, cis = [], [], []
    for key, values in data_to_plot.items():
        values = values[:,100:] # Ignore start of the training to decrease variance
        
        seed_means = np.mean(values, 1)
        seed_means = seed_means[~np.isnan(seed_means)]
        # Calculate 95% confidence interval
        confidence_interval = stats.sem(seed_means) * stats.t.ppf((1 + 0.95) / 2., len(seed_means) - 1)
        
        experiments.append(key)
        means.append(np.mean(seed_means))
        cis.append(confidence_interval)

    # Convert experiments to float for sorting and plotting
    experiments = [exp for exp in experiments]
    
    # Plotting
    plt.errorbar(experiments, means, yerr=cis, fmt='o', capsize=5, label="Mean ± 95% CI")
    plt.xlabel('Overshoot')
    plt.ylabel(f'Average {data_type}')
    plt.title(args.experiment_name)
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(source_dir, f"{data_type}.png"))
    plt.clf()


def comp2(file_name):
    if file_name == "baseline":
        return 0
    if file_name in ["nesterov", "nadam"]:
        return 0.0001
    if file_name.startswith("overshoot_"):
        return float(file_name.split("_")[-1])
    else:
        return 1000

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', type=str)
    args = parser.parse_args()


    source_dir = os.path.join('lightning_logs', args.experiment_name)
    distance_data = {}
    loss_data, loss_data_20, loss_data_100 = {}, {}, {}
    for job_name in sorted(os.listdir(source_dir), key=comp2):
        if not os.path.isdir(os.path.join(source_dir, job_name)):
            continue
        
        job_distance, job_loss, job_loss_20, job_loss_100 = [], [], [], []
        for version in os.listdir(os.path.join(source_dir, job_name)):
            stats_file_path = os.path.join(source_dir, job_name, version, 'training_stats.csv')
            if os.path.exists(stats_file_path):
                df = pd.read_csv(stats_file_path)
                job_distance.append(df['model_distance'].to_numpy())
                job_loss.append(df['base_loss'].to_numpy())
                # job_loss_20.append(df['base_loss_20'].to_numpy())
                job_loss_100.append(df['base_loss_100'].to_numpy())
        
        
        job_name = job_name.replace('overshoot_', '')
            
        distance_data[job_name] = np.array(job_distance)
        loss_data[job_name] = np.array(job_loss)
        # loss_data_20[job_name] = np.array(job_loss_20)
        loss_data_100[job_name] = np.array(job_loss_100)


    plot_data(distance_data, 'distance')
    plot_data(loss_data, 'loss')
    plot_loss_graph(loss_data_100, "loss_graph_100")
    print(f"Output generated in {source_dir}")
