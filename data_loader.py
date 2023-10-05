import torch
import numpy as np
import random

def worker_init_fix(worker_id):
    """
    Init dataloader workers with fixed seed.
    """
    random.seed(123 + worker_id)
    np.random.seed(123  + worker_id)


def worker_init_rand(worker_id):
    """
    Init dataloader workers with torch.initial_seed().
    torch.initial_seed() returns different seeds when called from different dataloader threads.
    """
    random.seed(torch.initial_seed())
    np.random.seed(torch.initial_seed() % 2 ** 32)


def get_dataloader(dataset, fix_seed=True, shuffle=False):
    return torch.utils.data.DataLoader(
        dataset, batch_size=32, shuffle=shuffle, num_workers= 1,
        pin_memory=False, worker_init_fn=worker_init_fix if fix_seed else worker_init_rand)