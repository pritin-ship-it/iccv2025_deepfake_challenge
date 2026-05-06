import os
import random
import torch
import wandb
import numpy as np
import datetime


def init_wandb(config):
    now = datetime.datetime.now()
    ts = now.strftime("%d.%m.%Y %H:%M")
    name = f"{config.wandb['experiment_name']} from {ts}"
    project = config.wandb['project_name']
    wandb.init(project=project, config=config, name=name)
   

# def set_seed(seed=42):
#     np.random.seed(seed)
#     random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.use_deterministic_algorithms(True) 
#     torch.backends.cudnn.deterministic = True
#     os.environ['PYTHONHASHSEED'] = str(seed)


def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)