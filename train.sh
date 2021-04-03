#!/bin/bash
# append provided source file information
python preprocess_labels.py $1

# preprocess files and output images in two resolutions
python src/preprocess.py 128 $2 
python src/preprocess.py 224 $2 

# train
python train.py model=densenet201 comment="model-1" gpus=0 &
python train.py model=tf_efficientnet_b5_ns data.n_mels=224 data.spec_min_width=224 comment="model-6" gpus=1 &
wait

python train.py model=ig_resnext101_32x8d comment="model-3" gpus=0 &
python train.py model=ig_resnext101_32x8d model.epochs=100 model.batch_size=96 comment="model-4" gpus=1 &
wait

python train.py model=seresnext50_32x4d comment="model-2" gpus=0 &
python train.py model=seresnext50_32x4d model.batch_size=120 data.n_mels=224 data.spec_min_width=224 comment="model-5" gpus=1 &
wait

python train.py model=resnet50 gradient_clip_val=1 weighted_sampler=True comment="model-9" gpus=0 &
python train.py model=resnet50 gradient_clip_val=1 weighted_sampler=True comment="model-8" loss_fn="lsep_loss" gpus=1 &
wait

python train.py model=resnet50 model.epochs=300 gradient_clip_val=1 weighted_sampler=True comment="model-7" gpus=0