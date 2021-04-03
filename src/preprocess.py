import concurrent.futures
import os
import pickle
import sys
from functools import partial
from pathlib import Path
from typing import Callable, List, Tuple

import librosa
import librosa.display
import numpy as np
from omegaconf import OmegaConf, DictConfig
from tqdm.auto import tqdm


def parallel(func: Callable, arr: List, max_workers: int = None) -> List:
    """Parallel execution of `func` over a list"""
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as ex:
        results = []
        for result in tqdm(ex.map(func, arr), total=len(arr)):
            results.append(result)
    return results


def read_audio(
    pathname: Path,
    config: DictConfig,
    trim_long_data: bool = False,
    trim_silence: bool = False,
) -> np.ndarray:
    y, _ = librosa.load(str(pathname), sr=config.sampling_rate)

    # trim silence
    if trim_silence and len(y) > 0:  # workaround: zero length causes error
        y, _ = librosa.effects.trim(y)  # trim, top_db=default(60)

    # make length config.samples
    if len(y) > config.min_sample_size:  # long enough
        if trim_long_data:
            y = y[0 : 0 + config.min_sample_size]
    else:  # pad blank
        padding = config.samples - len(y)  # add padding at both ends
        offset = padding // 2
        y = np.pad(y, (offset, config.min_sample_size - len(y) - offset), "constant")

    return y


def audio_to_melspectrogram(
    audio: np.ndarray, config: DictConfig, three_chanels: bool = False
) -> np.ndarray:
    spectrogram = librosa.feature.melspectrogram(
        audio,
        sr=config.sampling_rate,
        n_mels=config.n_mels,
        hop_length=config.hop_length,
        n_fft=config.n_fft,
        fmin=config.fmin,
        fmax=config.fmax,
    )
    logmel = librosa.power_to_db(spectrogram).astype(np.float32)

    if three_chanels:
        return np.array(
            [
                logmel,
                librosa.feature.delta(logmel),
                librosa.feature.delta(logmel, order=2),
            ]
        )
    else:
        return logmel


def normalize(X: np.ndarray, mean: float = None, std: float = None) -> np.ndarray:
    mean = mean or X.mean()
    std = std or (X - X.mean()).std()
    return ((X - mean) / std).astype(np.float16)


def preprocess(f: Path, config: DictConfig) -> Tuple:
    audio = read_audio(f, config=config, trim_long_data=False)
    logmel = audio_to_melspectrogram(audio, config=config, three_chanels=True)
    return (f.stem, logmel)


if __name__ == "__main__":
    dimension = int(sys.argv[1])
    data_dir = Path(sys.argv[2])
    save_dir = Path("/wdata/")

    config = OmegaConf.load(f"./conf/data/{dimension}.yaml")
    folder = f"preprocess-mel-{config.duration}-{config.n_mels}-{config.spec_min_width}"
    os.makedirs(save_dir / folder, exist_ok=True)

    data = parallel(
        partial(preprocess, config=config), list(data_dir.glob("*.flac")), max_workers=8
    )
    data = {int(k): v for k, v in data}

    with open(save_dir / f"{folder}/data.pkl", "wb") as f:
        pickle.dump(data, f)
