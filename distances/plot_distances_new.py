
import argparse
import yaml
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', type=str)
    args = parser.parse_args()


    source_dir = os.path.join('../lightning_logs', args.experiment_name)
    data_to_plot = {}
    for job_name in os.listdir(source_dir):
        if not os.path.isdir(os.path.join(source_dir, job_name)):
            continue
        
        job_resutls = []
        for version in os.listdir(os.path.join(source_dir, job_name)):
            if version.startswith('version_'):
                df = pd.read_csv(os.path.join(source_dir, job_name, version, 'training_stats.csv'))
                job_resutls.append(np.mean(df['model_distance']).item())
        
        if job_name.startswith('overshoot_'):
            job_name = job_name.split('_')[-1]
        data_to_plot[job_name] = job_resutls







    # with open(f'processed_distances/{args.model}_{args.dataset}.yaml', 'r') as file:
    #     results = yaml.safe_load(file)

    # Prepare data for plotting
    experiments = []
    means = []
    cis = []  # Confidence intervals

    for key, values in data_to_plot.items():
        values = np.array([v for v in values if not np.isnan(v)])  # Filter out NaN values
        mean = np.mean(values)
        # Calculate 95% confidence interval
        confidence_interval = stats.sem(values) * stats.t.ppf((1 + 0.95) / 2., len(values) - 1)
        
        experiments.append(key)
        means.append(mean)
        cis.append(confidence_interval)

    # experiments = experiments[1:2] + experiments[3:-1] + [experiments[2], experiments[-1]]
    # means = means[1:2] + means[3:-1] + [means[2], means[-1]]
    # cis = cis[1:2] + cis[3:-1] + [cis[2], cis[-1]]
    
    # experiments = experiments[:2] + experiments[3:]
    # means = means[:2] + means[3:]
    # cis = cis[:2] + cis[3:]

    # Convert experiments to float for sorting and plotting
    experiments = [exp for exp in experiments]

    # Plotting
    plt.errorbar(experiments, means, yerr=cis, fmt='o', capsize=5, label="Mean ± 95% CI")
    plt.xlabel('Overshoot')
    plt.ylabel('Average distances')
    plt.title('Weighted average distance among models CNN, MNIST')
    plt.grid(True)
    plt.legend()
    plt.savefig(f"processed_distances/tmp.png")
    # plt.show()