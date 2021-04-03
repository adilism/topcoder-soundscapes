import os
import pickle
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from torchvision import transforms
from tqdm.auto import tqdm

from src.dataset import MelSpectrogramDataset
from src.lightning import MelSpectrogramModel, predict
from src.utils import seed_everything


def get_model_predictions(model_path: Path, gpus: int = 0, n_classes: int = 9) -> Tuple:
    print(model_path)
    res = {"model": model_path, "n_samples": n_samples}

    # checkpoints
    print(f'Found {len(os.listdir(model_path / "checkpoints"))} checkpoints')

    # config
    cfg = OmegaConf.load(model_path / ".hydra/config.yaml")
    batch_size = 2 * cfg.model.batch_size
    print(OmegaConf.to_yaml(cfg))

    # load data
    mel_path = Path(
        f"/wdata/preprocess-mel-{cfg.data.clip_duration}-{cfg.data.spec_min_width}-{cfg.data.n_mels}"
    )

    test = pickle.load(open(mel_path / "data.pkl", "rb"))
    config = OmegaConf.load(f"./conf/data/{cfg.data.spec_min_width}.yaml")

    if cfg.model.n_channels == 1:
        test = {k: v[0, :, :] for k, v in test.items()}
        config.train_mean, config.train_std = config.train_mean[0], config.train_std[0]

    transforms_dict = {
        "train": transforms.Compose(
            [transforms.Normalize(config.train_mean, config.train_std)]
        ),
        "test": transforms.Compose(
            [transforms.Normalize(config.train_mean, config.train_std)]
        ),
    }

    # test set
    test_idx = test.keys()

    test_ds = MelSpectrogramDataset(
        [test[i] for i in test_idx],
        transforms_dict["test"],
        config=config,
        labels=None,
        apply_spec_aug=False,
        n_samples=n_samples,
    )

    test_dl = torch.utils.data.DataLoader(
        test_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
    )

    device = torch.device(f"cuda:{gpus}")

    for fold_idx in tqdm(range(5)):
        res[fold_idx] = {"checkpoints": []}
        res[fold_idx]["test"] = np.zeros((len(test_idx), n_classes))

        # fix earlier configs
        cfg.model_type = cfg.model_type or "timm"
        cfg.loss_fn = cfg.loss_fn or "BCEWithLogitsLoss"

        # checkpoints
        cp_paths = [
            str(cp) for cp in (model_path / "checkpoints").glob(f"*fold-{fold_idx}*")
        ]
        for cp in cp_paths:
            res[fold_idx]["checkpoints"].append(cp)

            seed_everything(fold_idx)
            model = MelSpectrogramModel.load_from_checkpoint(
                cp, map_location=device, cfg=cfg
            )

            test_preds = predict(test_dl, model, gpus=gpus, num_classes=n_classes)
            test_preds_tta = np.mean(
                test_preds.reshape(n_samples, -1, n_classes), axis=0
            )
            res[fold_idx]["test"] += test_preds_tta / len(cp_paths)
    return test_idx, res


if __name__ == "__main__":
    n_samples = 20
    num_workers = 8
    N_GPU = 0
    geometric_mean = False

    data_dir = Path("/wdata/")
    fn = sys.argv[1]

    save_dir = data_dir / "models"

    models = [
        "densenet201-ep2-bs120-128-model-1",
        "ig_resnext101_32x8d-ep2-bs128-128-model-3",
        "ig_resnext101_32x8d-ep2-bs96-128-model-4",
        "resnet50-ep2-bs32-128-model-7",
        "resnet50-ep2-bs32-128-model-8",
        "resnet50-ep2-bs32-128-model-9",
        "resnet50-ep2-bs400-128-model-7",
        "seresnext50_32x4d-ep2-bs120-224-model-5",
        "seresnext50_32x4d-ep2-bs128-128-model-2",
        "tf_efficientnet_b5_ns-ep2-bs64-224-model-6",
    ]

    # check that trained checkpoints exist
    for model in models:
        if not (save_dir / model).exists():
            print(f"Model {model} not found: please run the training script first")

    classes_ = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]
    n_classes = len(classes_)

    # predict
    out = []
    for model in models:
        test_idx, preds = get_model_predictions(save_dir / model, gpus=N_GPU)
        preds = np.mean([preds[i]["test"] for i in range(5)], axis=0)
        out.append(preds)

    if geometric_mean:
        # take geometric mean
        test_preds = np.exp(np.mean([np.log(i) for i in out], axis=0))
    else:
        # take mean
        test_preds = np.mean(out, axis=0)

    # create submission file
    sub = pd.DataFrame(index=test_idx, data=test_preds, columns=classes_)

    sub.index = sub.index.astype("str").str.zfill(10)
    sub.to_csv(fn, header=False, index=True)
