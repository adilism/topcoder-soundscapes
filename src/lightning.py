from typing import Callable, Dict, List, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import timm
import torch
import torch.nn as nn
from omegaconf import DictConfig
from omegaconf.omegaconf import OmegaConf
from sklearn.metrics import log_loss
from torch.autograd import Variable
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader


# Facebook's implementation is used for Mixup. With its default parameters.
def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.4) -> Tuple:
    """Mixup data augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(
    criterion: Callable,
    pred: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def _positive_sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def _negative_sigmoid(x: np.ndarray) -> np.ndarray:
    # Cache exp so you won't have to calculate it twice
    exp = np.exp(x)
    return exp / (exp + 1)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid implementation
    Source: https://stackoverflow.com/a/64717799
    """
    positive = x >= 0
    res = np.empty_like(x)
    res[positive] = _positive_sigmoid(x[positive])
    res[~positive] = _negative_sigmoid(x[~positive])
    return res


def add_weight_decay(
    model: nn.Module, weight_decay: float = 1e-5, skip_list: Union[List, Tuple] = ()
) -> List[Dict]:
    """Helper function to not decay weights in BatchNorm layers
    Source: https://discuss.pytorch.org/t/weight-decay-in-the-optimizers-is-a-bad-idea-especially-with-batchnorm/16994/3
    """
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": weight_decay},
    ]


class TimeFrequencyPooling(nn.Module):
    """Max-pool over time dimension (columns) then average-pool over mel dimension (rows)"""

    def __init__(self, output_size: int = 1, max_pool_first: bool = True) -> None:
        super().__init__()
        self.output_size = output_size
        if max_pool_first:
            self.ts_pool = nn.AdaptiveMaxPool2d(output_size=(self.output_size, None))
            self.fq_pool = nn.AdaptiveAvgPool1d(self.output_size)
        else:
            self.ts_pool = nn.AdaptiveMaxPool2d(output_size=(None, self.output_size))
            self.fq_pool = nn.AdaptiveAvgPool1d(self.output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ts_pool(x)
        x = self.fq_pool(x.squeeze())
        return x.squeeze()


def get_model(cfg: DictConfig, n_classes: int, model_type: str = "timm") -> nn.Module:
    """Create image model from `timm` backbone
    model_type = "tfpool" for time-frequency pooling"""
    model = timm.create_model(
        cfg.model.name,
        num_classes=n_classes,
        in_chans=cfg.model.n_channels,
        pretrained=True,
        drop_rate=cfg.model.drop_rate,
    )

    if model_type == "timm":
        return model
    elif model_type == "tfpool":
        model.global_pool = TimeFrequencyPooling()
        return model
    else:
        raise ValueError(f"Wrong model type: {model_type}")


def lsep_loss(
    input: torch.Tensor, target: torch.Tensor, average: bool = False
) -> torch.Tensor:
    """Log-sum-exponent pairwise loss
    Source: Kaggle"""

    differences = input.unsqueeze(1) - input.unsqueeze(2)
    where_different = (target.unsqueeze(1) < target.unsqueeze(2)).float()

    exps = differences.exp() * where_different
    lsep = torch.log(1 + exps.sum(2).sum(1))

    if average:
        return lsep.mean()
    else:
        return lsep.sum()


class MelSpectrogramModel(pl.LightningModule):
    """Main class for building mel spectrogram image models"""

    def __init__(
        self,
        cfg: DictConfig,
        n_classes: int = 9,
        n_samples: int = 5,
        pos_weight: torch.Tensor = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.model = get_model(cfg, n_classes, model_type=cfg.model_type)
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.lr = cfg.model.lr
        self.pct_start = cfg.model.warmup_prop
        self.epochs = cfg.model.epochs
        self.mixup_p = cfg.model.mixup
        self.to_float = True
        if cfg.loss_fn == "CrossEntropyLoss":
            self.loss_fn = nn.CrossEntropyLoss(reduction="sum", weight=pos_weight)
            self.to_float = False
        elif cfg.loss_fn == "BCEWithLogitsLoss":
            self.loss_fn = nn.BCEWithLogitsLoss(reduction="sum", pos_weight=pos_weight)
        elif cfg.loss_fn == "lsep_loss":
            self.loss_fn = lsep_loss
        else:
            raise ValueError(f"Wrong loss_fn parameter: {cfg.loss_fn}")
        self.log_loss = 10 ** 5

        self.save_hyperparameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.model.forward(x)
        return res

    def configure_optimizers(self) -> Tuple[List, List]:
        no_wd_list = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        opt_params = add_weight_decay(
            self.model, weight_decay=1e-2, skip_list=no_wd_list
        )

        optimizer = torch.optim.Adam(opt_params, lr=self.lr)
        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.lr,
            pct_start=self.pct_start,
            steps_per_epoch=len(self.train_dataloader()),
            epochs=self.epochs,
        )
        #     scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.1, verbose=0)

        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

        return [optimizer], [scheduler]

    def training_step(self, batch: Tuple, batch_idx: int) -> torch.Tensor:
        x, y = batch

        if self.to_float:
            y = y.float()
        else:
            y = y.long()

        do_mixup = np.random.random() < self.mixup_p
        if do_mixup:
            x, y_a, y_b, lam = mixup_data(x, y)
            x, y_a, y_b = map(Variable, (x, y_a, y_b))

        y_pred = self.model(x).view(-1, self.n_classes)

        if do_mixup:
            loss = mixup_criterion(self.loss_fn, y_pred, y_a, y_b, lam)
        else:
            loss = self.loss_fn(y_pred, y)

        self.log("train_loss", loss / y_pred.size(0))
        return loss

    def validation_step(self, batch: Tuple, batch_idx: int) -> Dict:
        x, y = batch

        if self.to_float:
            y = y.float()
        else:
            y = y.long()

        y_pred = self.model(x).view(-1, self.n_classes)
        val_loss = self.loss_fn(y_pred, y)

        val_loss = (val_loss / y_pred.size(0)).item()
        self.log("val_loss", val_loss, on_epoch=True)
        return {
            "val_loss": val_loss,
            "labels": y.cpu().numpy(),
            "preds": sigmoid(y_pred.cpu().numpy()),
        }

    def validation_epoch_end(self, outputs: Dict) -> None:
        preds = np.concatenate([output["preds"] for output in outputs], axis=0)
        y = np.concatenate([output["labels"] for output in outputs], axis=0)

        preds_tta = np.mean(preds.reshape(self.n_samples, -1, self.n_classes), axis=0)
        y_tta = np.mean(y.reshape(self.n_samples, -1, self.n_classes), axis=0)

        try:
            self.log_loss = log_loss(y_tta, preds_tta)
        except ValueError:
            self.log_loss += 1e-5

        self.log("logloss", torch.tensor(self.log_loss))
        return


def predict(
    dl: DataLoader,
    model: Union[nn.Module, pl.LightningModule],
    gpus: str = "0",
    num_classes: int = 9,
) -> np.ndarray:
    device = torch.device("cpu") if gpus == "cpu" else torch.device(f"cuda:{gpus}")
    model.to(device)
    model.eval()
    res = []
    with torch.no_grad():
        for x, target in dl:
            preds = model(x.to(device)).view(-1, num_classes).cpu().detach()
            preds = sigmoid(preds.numpy())
            res.append(preds)
    return np.concatenate(res, axis=0)
