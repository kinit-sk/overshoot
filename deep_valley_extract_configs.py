import os
import json


task_name_map = {
    "mnist_vae": "mnist_vae",
    "fmnist_vae": "fmnist_vae",
    "fmnist_2c2d": "fmnist_2c2d",
    "cifar10_3c3d": "cifar_3c3d",
}


# These are from https://github.com/fsschneider/DeepOBS/tree/master/deepobs/tensorflow/testproblems
# According to paper (https://arxiv.org/pdf/2007.01547), weight-decay wasn't finetuned
task_weight_decay_map = {
    "mnist_vae": 0,
    "fmnist_vae": 0,
    "fmnist_2c2d": 0,
    "cifar10_3c3d": 0.002,
}


optimizer_name_map = {
    "MomentumOptimizer": "sgd",
    "NAGOptimizer": "nesterov",
    "AdamOptimizer": "adam",
    "NadamOptimizer": "nadam",
}


def string_conf(task_name: str, optimizer: str, setting: dict):
    conf =  f"""
@dataclass
class LargeBudget__CosineScheduler__{task_name_map[task_name]}__{optimizer_name_map[optimizer]}__Config(DefaultConfig):
    use_lr_scheduler: bool = True
    epochs: int = {setting['epochs']}
    B: int = {setting['batch_size']}
    lr: float = {setting['optimizer_hyperparams']['learning_rate']}
    weight_decay: float = {task_weight_decay_map[task_name]}"""

    if 'beta1' in setting['optimizer_hyperparams']:
        conf += f"""
    beta1: float = {setting['optimizer_hyperparams']['beta1']}
    beta2: float = {setting['optimizer_hyperparams']['beta2']}
    epsilon: float = {setting['optimizer_hyperparams']['epsilon']}
"""
    elif 'momentum' in setting['optimizer_hyperparams']:
        conf += f"""
    momentum: float = {setting['optimizer_hyperparams']['momentum']}
"""
    else:
        raise Exception(f'Optimizer hyperparams are missing values')
    return conf

def process_json(data):
    min_loss = min(data['test_losses'])
    setting = {
        "epochs": data['num_epochs'],
        "batch_size": data['batch_size'],
        "optimizer_hyperparams": data['optimizer_hyperparams'],
    }
    return min_loss, setting


if __name__ == "__main__":
    
    # Path to this repo: https://github.com/SirRob1997/Crowded-Valley---Results
    path_to_repo = "/home/kopi/kinit/Crowded-Valley---Results"
    budget = "large_budget"
    scheduler = "cosine"
    tasks = ["mnist_vae", "fmnist_vae", "fmnist_2c2d", "cifar10_3c3d"]
    optimizers = ["AdamOptimizer", "MomentumOptimizer", "NadamOptimizer", "NAGOptimizer"]
    root = os.path.join(path_to_repo, "results_main", budget, scheduler)

    for task in tasks:
        for optimizer in optimizers:
            task_optimizer_root = os.path.join(root, task, optimizer)
            min_loss, min_setting, best_setting = float("inf"), "", {}
            for setting_name in os.listdir(task_optimizer_root):
                name = os.listdir(os.path.join(task_optimizer_root, setting_name))[0]
                with open(os.path.join(task_optimizer_root, setting_name, name), "r") as f:
                    loss, setting = process_json(json.load(f))
                if loss < min_loss:
                    min_loss = loss
                    min_setting = setting_name
                    best_setting = setting
            print(string_conf(task, optimizer, best_setting))