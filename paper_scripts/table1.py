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
    "adam_overshoot_7": "Adam7",
    "adam_overshoot_adaptive": "AdamA",
}

task_name_mapping = {
    "mlp_housing": "Housing",
    "mlp_mnist": "Mnist",
    "2c2d_fashion": "Fashion",
}

task_name_to_min_max = {
    "Housing": min,
    "Mnist": max,
    "Fashion": max
}


rows_to_drop = [
    "Mnist"
]
columns_to_drop = [
    "SGDA",
    "AdamA",
]

def bold_min(row):
    # import code; code.interact(local=locals())
    fn_to_use = task_name_to_min_max[row.name]
    sgd_min = fn_to_use([x[1][0] for x in row.items() if 'SGD' in x[0] or 'CM' in x[0]])
    adam_min = fn_to_use([x[1][0] for x in row.items() if 'Adam' in x[0]])
    return [f"\\textbf{{{val[0]} \u00B1{val[1]}}}" if (val[0] == sgd_min or val[0] == adam_min) else f"{val[0]} \u00B1{val[1]}" for _, val in row.items()]


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

            if task_name == "mlp_housing":
                resutls = [x * 100 for x in resutls]

            task_results[optimizers_names_mapping[run_name]] = mean_confidence_interval(resutls)

        
        all_results[task_name] = task_results
            
    # print(all_results)
    df = pd.DataFrame(all_results).T
    df = df[list(optimizers_names_mapping.values())]
    df = df.rename(index=task_name_mapping)
    df.drop(rows_to_drop, inplace=True)
    df.drop(columns_to_drop, axis=1, inplace=True)
    

    # Apply the function to each row and create a new DataFrame
    bolded_df = pd.DataFrame([bold_min(row) for _, row in df.iterrows()], columns=df.columns, index=df.index)

    # Convert to LaTeX table with column names, ensuring escape=False
    latex_table = bolded_df.to_latex(escape=False, column_format='l' + 'p{0.7cm}' * len(df.columns))
    # print(type(latex_table))

    print(latex_table)



    # pd.read_csv()