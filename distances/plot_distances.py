import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--dataset', type=str)
    args = parser.parse_args()




    with open(f'processed_distances/{args.model}_{args.dataset}.yaml', 'r') as file:
        results = yaml.safe_load(file)

    # Prepare data for plotting
    experiments = []
    means = []
    cis = []  # Confidence intervals

    for key, values in results.items():
        values = np.array([v for v in values if not np.isnan(v)])  # Filter out NaN values
        mean = np.mean(values)
        # Calculate 95% confidence interval
        confidence_interval = stats.sem(values) * stats.t.ppf((1 + 0.95) / 2., len(values) - 1)
        
        experiments.append(key)
        means.append(mean)
        cis.append(confidence_interval)

    # Convert experiments to float for sorting and plotting
    experiments = [exp for exp in experiments]

    # Plotting
    plt.errorbar(experiments, means, yerr=cis, fmt='o', capsize=5, label="Mean ± 95% CI")
    plt.xlabel('Overshoot')
    plt.ylabel('Average distances')
    plt.title('Weighted average distance among models')
    plt.grid(True)
    plt.legend()
    plt.savefig(f"processed_distances/{args.model}_{args.dataset}.png")
    # plt.show()