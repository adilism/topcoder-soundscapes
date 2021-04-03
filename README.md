# Topcoder Soundscapes Marathon Match Academic Prize solution

This is a solution to the [Soundscapes Marathon Match](https://www.topcoder.com/challenges/8440570d-d16b-43df-be4e-a720577626d5?tab=details) competition organised by Topcoder and National Geospatial-Intelligence Agency. The main goal of the competition was to geo-locate audio files consisting of non-speech ambient noise in one of nine target cities.

# Solution
The final solution consists of an ensemble of nine CNN models fitted on mel spectrograms, with heavy use of data augmentations.

Key points:
- Raw audio converted to log-scaled mel spectrograms
- Random crops of mel spectrograms along time axis as input during training
- Five-fold group cross-validation
- Pretrained CNN models from [timm](https://github.com/rwightman/pytorch-image-models)
- SpecAugment and Mixup data augmentation techniques

# Replicate results
The results were obtained using two Nvidia V100 GPUs, however the training script can be adapted to a different GPU setup. The code requires Docker and [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) installed.

## Steps
1. Build a Docker image from the Dockerfile
2. Run the container. The training data expected to be in `/data` and output will be saved in `/wdata`. For instance, run

```
docker run --gpus all --ipc=host \
	-v ~/<path>/data/train:/data \
    -v ~/<path>/wdata:/wdata \
    -it <image-name>
```

3. Run `train.sh <path-to-speech-files> <path-to-files-without-speech>` to preprocess .flac files and fit the models. This should create 9 folders in `/wdata/models` containing model checkpoints
4. Run `test.sh <path-to-files-without-speech> <path-to-output-file>` to preprocess test .flac files and make predictions.