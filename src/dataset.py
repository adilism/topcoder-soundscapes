import random
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from torch.utils.data.dataloader import DataLoader
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from torch.utils.data import Dataset
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms


def spec_augment(
    spec: np.ndarray,
    num_mask: int = 2,
    freq_masking_max_percentage: float = 0.15,
    time_masking_max_percentage: float = 0.15,
) -> np.ndarray:
    """SpecAugment data augmentation: mask random bands over frequency and time domains"""
    spec = spec.copy()

    for i in range(num_mask):
        _, num_freqs, num_frames = spec.shape
        freq_percentage = random.uniform(0.0, freq_masking_max_percentage)
        time_percentage = random.uniform(0.0, time_masking_max_percentage)

        num_freqs_to_mask = int(freq_percentage * num_freqs)
        num_frames_to_mask = int(time_percentage * num_frames)

        t0 = int(np.random.uniform(low=0.0, high=num_frames - num_frames_to_mask))
        f0 = int(np.random.uniform(low=0.0, high=num_freqs - num_freqs_to_mask))

        spec[:, :, t0 : t0 + num_frames_to_mask] = 0
        spec[:, f0 : f0 + num_freqs_to_mask, :] = 0

    return spec


class MelSpectrogramDataset(Dataset):
    def __init__(
        self,
        mels: List,
        transforms: transforms.Compose,
        config: DictConfig,
        apply_spec_aug: bool = False,
        labels: np.ndarray = None,
        n_samples: int = 1,
    ):
        super().__init__()
        self.mels = mels
        self.transforms = transforms
        self.apply_spec_aug = apply_spec_aug
        self.n_samples = n_samples
        self.config = config
        self.labels = np.zeros(len(self.mels)) if labels is None else labels

    def __len__(self):
        return len(self.mels) * self.n_samples

    def __getitem__(self, idx: int):
        idx = idx % len(self.mels)

        data = self.mels[idx].astype(np.float32)
        _, _, time_dim = data.shape

        crop = random.randint(0, time_dim - self.config.spec_min_width)
        data = data[:, :, crop : crop + self.config.spec_min_width]

        if self.apply_spec_aug:
            data = spec_augment(data)

        data = torch.from_numpy(data)
        data = self.transforms(data)

        label = torch.as_tensor(self.labels[idx]).float()
        return data, label


def get_dataloaders(
    train: Dict,
    labels: pd.DataFrame,
    lb: Union[LabelBinarizer, LabelEncoder],
    batch_size: int,
    transforms_dict: Dict,
    config: DictConfig,
    train_idx: List,
    valid_idx: List,
    n_samples: int,
    apply_spec_aug: bool = True,
    sample_weights: List = None,
    num_workers: int = 8,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    train_ds = MelSpectrogramDataset(
        [train[i] for i in train_idx],
        transforms_dict["train"],
        config=config,
        labels=lb.transform(labels.loc[train_idx, "labels"]),
        apply_spec_aug=apply_spec_aug,
    )

    valid_ds = MelSpectrogramDataset(
        [train[i] for i in valid_idx],
        transforms_dict["test"],
        config=config,
        labels=lb.transform(labels.loc[valid_idx, "labels"]),
        apply_spec_aug=False,
        n_samples=n_samples,
    )

    valid_dl = torch.utils.data.DataLoader(
        valid_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=pin_memory,
    )

    if sample_weights is None:
        train_dl = torch.utils.data.DataLoader(
            train_ds,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            pin_memory=pin_memory,
        )

    else:
        train_sampler = WeightedRandomSampler(
            weights=[sample_weights[i] for i in train_idx],
            num_samples=len(train_idx),
            replacement=True,
        )

        train_dl = torch.utils.data.DataLoader(
            train_ds,
            batch_size=batch_size,
            num_workers=num_workers,
            sampler=train_sampler,
            pin_memory=pin_memory,
        )
    return train_dl, valid_dl
