import argparse
import os

import pandas as pd
import numpy as np
import scipy.stats as stats





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
    "mlp_housing": "MLP-CA",
    "vae_mnist": "VAE-M",
    "vae_f-mnist": "VAE-FM",
    "mlp_mnist": "Mnist",
    "2c2d_fashion": "2c2d-FM",
    "3c3d_cifar10": "3c3d-C10",
    "resnet18_cifar100": "ResNet-C100 old",
    "resnet18_cifar100_v2_better_aug_weight_decay_new_sgd": "ResNet-C100",
}

rows_to_drop = [
    "Mnist",
    "ResNet-C100 old"
]
columns_to_drop = [
    "SGDA",
    "AdamA",
]

def is_classification(name):
    return ("VAE" not in name) and ("MLP" not in name)
    
def is_sgd(name):
    return ("SGD" in name) or ("CM" in name) or ("NAG" in name)


def process_sub_row(row_name, sub_row):
    fn_to_use = max if is_classification(row_name) else min

    means = [mean_confidence_interval(values) for _, values in sub_row]
    best_value = fn_to_use(means, key=lambda x: x[0])[0]
    better_baseline = fn_to_use([(values, np.mean(values)) for _, values in sub_row[:2]], key=lambda x: x[1])[0]
    
    reject_same_dist = [False, False]
    for _, overshoot_values in sub_row[2:]:
        alpha = 0.05
        better_baseline = better_baseline[:min(len(better_baseline), len(overshoot_values))]
        overshoot_values = overshoot_values[:min(len(better_baseline), len(overshoot_values))]
        
        test = stats.ttest_rel(better_baseline, overshoot_values)
        reject_same_dist.append(test.pvalue < alpha)

    ps = lambda use_star: "*" if use_star else ""
    return [f"\\textbf{{{mean:.2f}{ps(use_star)} \u00B1{interval:.2f}}}" if mean == best_value else f"{mean:.2f}{ps(use_star)} \u00B1{interval:.2f}" for (mean, interval), use_star in zip(means, reject_same_dist)]
    
def process_row(row):
    sgd_sub_row = process_sub_row(row.name, [item for item in row.items() if is_sgd(item[0])])
    adam_sub_row = process_sub_row(row.name, [item for item in row.items() if not is_sgd(item[0])])
    sgd_sub_row.extend(adam_sub_row)
    return sgd_sub_row

def add_multirow(table: str) -> str:
    table = table.split('\n')
    table[4] = table[4].replace("MLP", "\\multirow{3}{*}{\\vspace*{-1.0cm} Loss} & MLP").replace(r'\\', r'\\ \cline{2-12}')
    table[5] = table[5].replace("VAE", "& VAE").replace(r'\\', r'\\ \cline{2-12}')
    table[6] = table[6].replace("VAE", "& VAE").replace(r'\\', r'\\ \cline{1-12}')
    
    table[7] = table[7].replace("2c2d", "\\multirow{3}{*}{\\vspace*{-1.0cm} Acc} & 2c2d").replace(r'\\', r'\\ \cline{2-12}')
    table[8] = table[8].replace("3c3d", "& 3c3d").replace(r'\\', r'\\ \cline{2-12}')
    table[9] = table[9].replace("Res", "& Res")
    return '\n'.join(table)



def mean_confidence_interval(data, confidence_interval: float=0.95):
    mean = np.mean(data)

    # Calculate standard error of the mean
    sem = stats.sem(data)

    # Calculate the 95% confidence interval
    confidence_interval = stats.t.interval(confidence_interval, len(data) - 1, loc=mean, scale=sem)

    return round(mean, 2), round(mean - confidence_interval[0], 2)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    args = parser.parse_args()


    root = "../lightning_logs/table1"

    all_results = {}
    for task_name in os.listdir(root):
        task_root = os.path.join(root, task_name)

        task_results = {}
        for run_name in os.listdir(task_root):
            run_root = os.path.join(task_root, run_name)
            if not os.path.isdir(run_root):
                continue
            
            resutls = []
            for seeds in os.listdir(run_root):
                finel_path = os.path.join(run_root, seeds)
            
                if "housing" in task_name:
                    validation_stats = os.path.join(finel_path, "validation_stats.csv")
                else:
                    validation_stats = os.path.join(finel_path, "test_stats.csv")
                if not os.path.exists(validation_stats):
                    continue
                
                df = pd.read_csv(validation_stats)
                if "accuracy" in df.columns:
                    resutls.append(df["accuracy"].max())
                else:
                    resutls.append(df["loss"].min())

            if len(resutls)> 1:
                if task_name == "mlp_housing":
                    resutls = [x * 100 for x in resutls]
                # task_results[optimizers_names_mapping[run_name]] = mean_confidence_interval(resutls)
                task_results[optimizers_names_mapping[run_name]] = resutls

        if len(task_results):
            all_results[task_name] = task_results
            
    # print(all_results)
    df = pd.DataFrame(all_results).T
    df = df[list(optimizers_names_mapping.values())]
    df = df.rename(index=task_name_mapping)
    df = df.reindex(index=task_name_mapping.values())

    df.drop(rows_to_drop, inplace=True)
    df.drop(columns_to_drop, axis=1, inplace=True)


    # Apply the function to each row and create a new DataFrame
    processed = pd.DataFrame([process_row(row) for _, row in df.iterrows()], columns=df.columns, index=df.index)

    # Convert to LaTeX table with column names, ensuring escape=False
    latex_table = processed.to_latex(escape=False, column_format='l' + 'p{0.7cm}' * len(df.columns))


    latex_table = add_multirow(latex_table)
    # latex_table = latex_table.replace(r'\\', r'\\ \hline')

    print(latex_table)



    # pd.read_csv()