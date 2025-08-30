from typing import List

import numpy as np
import torch

from training.dataset import Dataset


class ClassSampler:
    def __init__(self, dataset: Dataset, ood: bool = True) -> None:
        super().__init__()
        self.dt = dataset
        self.ood = ood    # Whether to sample classes uniformly (CGANs only)

    def sample_y(self, bs: int, device_bs: int, device: torch.device = torch.device("cpu")) -> List[List[torch.Tensor]]:
        labels = self._get_labels(bs)
        labels = torch.from_numpy(np.stack(labels)).pin_memory().to(device)
        labels = labels.split(device_bs)
        return labels

    def sample_z(self, z_dim: int, bs: int, device_bs: int, device: torch.device("cpu")):
        z = torch.randn([bs, z_dim], device=device)
        z = z.split(device_bs)
        return z

    def _get_labels(self, n: int):
        if self.ood:
            labels = np.random.randint(0, self.dt.label_dim, size=(n,))
            onehot = np.zeros((n, self.dt.label_dim), dtype=np.float32)
            onehot[np.arange(n), labels] = 1
            return onehot

        return [self.dt.get_label(np.random.randint(len(self.dt))) for _ in range(n)]


class TwinsClassSampler(ClassSampler):
    def sample_y(self, bs: int, device_bs: int, device: torch.device = torch.device("cpu")) -> List[List[torch.Tensor]]:
        labels = super().sample_y(bs // 2, device_bs // 2, device)
        labels = [lbl_device.repeat(2, *tuple([1 for _ in range(len(lbl_device.shape) - 1)])) for lbl_device in labels]
        return labels
    
    def sample_z(self, z_dim: int, bs: int, device_bs: int, device: torch.device = torch.device("cpu")) -> List[List[torch.Tensor]]:
        z = super().sample_z(z_dim, bs // 2, device_bs // 2, device)
        z = [z_device.repeat(2, *tuple([1 for _ in range(len(z_device.shape) - 1)])) for z_device in z]
        return z