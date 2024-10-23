import argparse
import os
import yaml
import numpy as np
from tqdm import tqdm



def compute_distance(models_a, models_b, momentum = 0.9):

    steps = models_a.shape[0]
    summ = 0
    for i in range(steps):
        for j in range(i):
            summ += np.linalg.norm(models_a[i] - models_b[j]) * (momentum**(i - j))

    return (summ / steps).item()

def load_models(name):
    with open(f'raw_distances/{name}', 'rb') as f:
        return np.load(f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--dataset', type=str)
    # parser.add_argument('--seed', type=str)
    args = parser.parse_args()


    distances = {}
    for name in tqdm(os.listdir('raw_distances')):

        if not name.endswith('.npy'):
            continue
        
        t = name.split('_')[-2]
        if t not in distances:
            distances[t] = []
            
        if name.startswith(f'baseline_base_models_{args.model}_{args.dataset}'):
            baseline_base_models = load_models(name)
            distances[t].append(compute_distance(baseline_base_models, baseline_base_models))
        elif name.startswith(f'overshoot_base_models_{args.model}_{args.dataset}'):
            overshoot_base_models = load_models(name)
            overshoot_overshoot_models = load_models(f'overshoot_overshoot_{name[15:]}')
            distances[t].append(compute_distance(overshoot_base_models, overshoot_overshoot_models))

        print(distances)
        print("====================")
            

    with open(f'processed_distances/{args.model}_{args.dataset}.yaml', 'w') as f:
        yaml.dump(distances, f, default_flow_style=False)
        