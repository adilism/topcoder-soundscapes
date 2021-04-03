import logging
import pickle
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from torchvision import transforms

from src.dataset import get_dataloaders
from src.lightning import MelSpectrogramModel
from src.utils import get_group_folds, seed_everything


@hydra.main(config_path="conf", config_name="config")
def run(cfg: DictConfig) -> None:
    logging.info(OmegaConf.to_yaml(cfg))

    # folds
    n_samples = 5
    n_splits = 5
    n_repeats = 1

    n_workers = 8
    early_stop = False
    save_dir = Path(".")  # working directory is managed by hydra

    DATA_DIR = Path("/wdata/")

    # process labels to work with the selected loss function
    labels = pickle.load(open(DATA_DIR / "train-groups.pkl", "rb"))

    if cfg.loss_fn in ["FocalLoss", "CrossEntropyLoss"]:
        lb = LabelEncoder()
    else:
        lb = LabelBinarizer()

    lb.fit(labels["labels"])
    n_classes = len(lb.classes_)

    mel_path = f"preprocess-mel-{cfg.data.clip_duration}-{cfg.data.spec_min_width}-{cfg.data.n_mels}"
    if cfg.speech:
        mel_path = mel_path + "-speech"
    mel_path = DATA_DIR / mel_path

    with open(mel_path / "data.pkl", "rb") as f:
        train = pickle.load(f)

    config = OmegaConf.load(f"../../../work/conf/data/{cfg.data.spec_min_width}.yaml")

    if cfg.model.n_channels == 1:
        train = {k: v[0, :, :] for k, v in train.items()}
        config.train_mean, config.train_std = config.train_mean[0], config.train_std[0]

    transforms_dict = {
        "train": transforms.Compose(
            [
                transforms.Normalize(config.train_mean, config.train_std),
            ]
        ),
        "test": transforms.Compose(
            [
                transforms.Normalize(config.train_mean, config.train_std),
            ]
        ),
    }

    folds = get_group_folds(
        labels, n_splits=n_splits, n_repeats=n_repeats, stratified=True
    )

    # overweight underrepresented classes, assumes classes are encoded {A:0, B:1, etc.}
    equal_cls_weights = (
        labels["labels"].value_counts().apply(lambda x: 56 / x).sort_index()
    )

    pos_weight = torch.from_numpy(equal_cls_weights.values) if cfg.pos_weight else None

    if cfg.weighted_sampler:
        k = 0.5
        logging.info(f"Sample weights: {equal_cls_weights.pow(k).values} with k={k}")
        sample_weights = labels["labels"].map(equal_cls_weights.pow(k))
    else:
        sample_weights = None

    for fold_idx in cfg.folds:
        print(f"Fold {fold_idx}")
        train_idx, valid_idx = folds[fold_idx]

        train_dl, valid_dl = get_dataloaders(
            train,
            labels,
            lb,
            cfg.model.batch_size,
            transforms_dict,
            config,
            train_idx,
            valid_idx,
            n_samples,
            apply_spec_aug=True,
            sample_weights=sample_weights,
            num_workers=n_workers,
            pin_memory=False,
        )

        model = MelSpectrogramModel(
            cfg=cfg, n_classes=n_classes, n_samples=n_samples, pos_weight=pos_weight
        )

        callbacks = []

        lr_monitor = LearningRateMonitor(logging_interval="step")
        callbacks.append(lr_monitor)

        metric_name, metric_mode = "logloss", "min"

        if early_stop:
            early_stop_callback = EarlyStopping(
                monitor=metric_name,
                min_delta=0.00,
                patience=10,
                verbose=False,
                mode=metric_mode,
            )
            callbacks.append(early_stop_callback)

        checkpoint_callback = ModelCheckpoint(
            verbose=False,
            monitor=metric_name,
            dirpath=save_dir / "checkpoints",
            filename="{epoch:02d}",
            prefix=f"fold-{fold_idx}",
            save_last=True,
            save_top_k=3,
            mode=metric_mode,
        )
        callbacks.append(checkpoint_callback)

        trainer = pl.Trainer(
            num_sanity_val_steps=0,
            default_root_dir=".",
            accumulate_grad_batches=cfg.model.accumulate_grad_batches,
            gradient_clip_val=cfg.gradient_clip_val,
            gpus=str(cfg.gpus),
            precision=16,
            callbacks=callbacks,
            max_epochs=cfg.model.epochs,
        )

        seed_everything(fold_idx)
        trainer.fit(model, train_dl, valid_dl)


if __name__ == "__main__":
    run()
