"""Continual learning benchmark datasets.

Provides two standard benchmarks built on top of MNIST:

* SplitMNIST: the ten digit classes are divided into five consecutive
  tasks with two classes each (0/1, 2/3, ..., 8/9).
* PermutedMNIST: every task uses all ten classes, but a different fixed
  random permutation is applied to the 784 input pixels.

Both benchmarks yield a sequence of (train, test) DataLoader pairs, one per
task, which is exactly the setting in which catastrophic forgetting occurs.
"""

import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")


def _load_mnist(data_dir=DATA_DIR):
    """Loads the raw MNIST train/test splits as tensors."""
    tf = transforms.Compose([transforms.ToTensor()])
    train = datasets.MNIST(data_dir, train=True, download=False, transform=tf)
    test = datasets.MNIST(data_dir, train=False, download=False, transform=tf)
    return train, test


def _flatten(dataset):
    """Converts an MNIST dataset into flat (x, y) float tensors."""
    xs = torch.stack([dataset[i][0] for i in range(len(dataset))])
    ys = torch.tensor([dataset[i][1] for i in range(len(dataset))])
    return xs.view(len(dataset), -1).float(), ys.long()


def make_split_mnist(n_tasks=5, batch_size=256, data_dir=DATA_DIR):
    """Builds the SplitMNIST benchmark.

    Task t contains the digits {2t, 2t+1}.  Each task keeps its own two-way
    output head (task-incremental protocol).
    """
    train, test = _load_mnist(data_dir)
    xtr, ytr = _flatten(train)
    xte, yte = _flatten(test)

    tasks = []
    for t in range(n_tasks):
        classes = [2 * t, 2 * t + 1]
        tr_idx = [i for i in range(len(ytr)) if int(ytr[i]) in classes]
        te_idx = [i for i in range(len(yte)) if int(yte[i]) in classes]
        # Remap global labels (2t, 2t+1) to local head labels (0, 1).
        ytr_local = (ytr[tr_idx] - 2 * t)
        yte_local = (yte[te_idx] - 2 * t)
        tr = TensorDataset(xtr[tr_idx], ytr_local)
        te = TensorDataset(xte[te_idx], yte_local)
        tasks.append((
            DataLoader(tr, batch_size=batch_size, shuffle=True, drop_last=True),
            DataLoader(te, batch_size=1024, shuffle=False),
            len(classes),
        ))
    return tasks


class _PermutedDataset:
    """Memory-efficient PermutedMNIST task dataset.

    Instead of materialising a permuted copy of the whole data set for every
    task (about 190 MB per task), the permutation is applied on the fly in
    ``__getitem__``; the base tensors are stored only once and shared.
    """

    def __init__(self, x, y, perm):
        self.x = x
        self.y = y
        self.perm = perm

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx][self.perm], self.y[idx]


def make_permuted_mnist(n_tasks=5, batch_size=256, seed=0, data_dir=DATA_DIR):
    """Builds the PermutedMNIST benchmark.

    Every task applies its own fixed random pixel permutation; all ten output
    classes are shared, so a single ten-way head is used.
    """
    train, test = _load_mnist(data_dir)
    xtr, ytr = _flatten(train)
    xte, yte = _flatten(test)

    rng = np.random.RandomState(seed)
    tasks = []
    for t in range(n_tasks):
        if t == 0:
            perm = np.arange(784)          # first task = original MNIST
        else:
            perm = rng.permutation(784)
        perm = torch.from_numpy(perm).long()
        tr = _PermutedDataset(xtr, ytr, perm)
        te = _PermutedDataset(xte, yte, perm)
        tasks.append((
            DataLoader(tr, batch_size=batch_size, shuffle=True, drop_last=True),
            DataLoader(te, batch_size=1024, shuffle=False),
            10,
        ))
    return tasks
