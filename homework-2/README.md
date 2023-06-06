# Homework 2: Fruit classification based on ResNet50

[[Description]](https://github.com/open-mmlab/OpenMMLabCamp/issues/ )

[[Code Base]](https://github.com/open-mmlab/mmpretrain)

[[Data]](https://drive.google.com/file/d/1-6cz8pL5LNk14vXMRDNkGJMhKWJrhi8W/view?usp=sharing)

[[Video]](https://www.bilibili.com/video/BV1Ju4y1Z7ZE)

## MMPreTrain

For example, `resnet18_8xb32_in1k.py` is the config file for the `ResNet18` model, where `8xb32` means 8 GPUs and batch size 32 each, `in1k` means the model is designed on ImageNet-1K dataset. The config file is organized as follows:

```
# mmpretrain/configs/resnet/resnet18_8xb32_in1k.py

_base_ = [
    '../_base_/models/resnet18.py', '../_base_/datasets/imagenet_bs32.py',
    '../_base_/schedules/imagenet_bs256.py', '../_base_/default_runtime.py'
]
```

The four config python files are corrspoinding to the four dir/files in the `mmpretrain/configs/_base_/` folder.

```
mmpretrain/configs/_base_/
├── datasets/
├── models/
├── schedules/
├── default_runtime.py
```

## Environment Setup

1. Install MMPreTrain
2. Copy `resnet50` config files to the code/ and do some modifications
3. Why choose `resnet50_8xb32....in1k.py`? Because fruit datasets contain 30 classes which is suitable for batchsize of 32 and the resoluation of each image is around 500x500. And we have 8 candidates:

```
resnet50_8xb32-coslr_in1k.py
resnet50_8xb32-fp16-dynamic_in1k.py
resnet50_8xb32-lbs_in1k.py
resnet50_8xb32-coslr-preciseBN_in1k.py
resnet50_8xb32-fp16_in1k.py
resnet50_8xb32-mixup_in1k.py
resnet50_8xb32-cutmix_in1k.py
resnet50_8xb32_in1k.py
```

`in1k.py` is a baseline, `coslr`, `lbs` (label smooth), `mixup`, `cutmix` `preciseBN` are some tricks to improve the performance. `fp16` is a trick to speed up the training process. We simply choose `resnet50_8xb32_in1k.py` as our base config file.

Code file organization:

```
code/
├── configs/
│   ├── _base_/
|   |   ├── datasets/imagenet_bs32.py
|   |   ├── models/resnet50.py
|   |   ├── schedules/imagenet_bs256.py
|   |   ├── default_runtime.py
│   ├── resnet50_8xb32_in1k.py
├── mmpretrain/
```

## Split Dataset

train:val = 8:2

```
cd code

python split_dataset.py
```

## Train

```
mim train mmpretrain configs/resnet50_8xb32_fruit30.py --work-dir=../log/train
```

[homework-2/log/train/20230606_164304/vis_data/20230606_164304.json](log/train/20230606_164304/vis_data/20230606_164304.json)

## Test

```
mim test mmpretrain configs/resnet50_8xb32_fruit30.py --checkpoint=../log/train/epoch_100.pth --work-dir=../log/test
```

## Results

All parameters are default in `resnet50_8xb32_in1k.py` -> 60.6982%

[homework-2/log/test/20230606_170156/20230606_170156.json](log/test/20230606_170156/20230606_170156.json)

```
{"accuracy/top1": 60.69819641113281, "accuracy/top5": 91.66666412353516, "data_time": 0.0024173736572265627, "time": 0.01637766361236572}
```

Since we finetune a pretrained model, we set a smaller learning rate and do not need too many epochs.

```
lr: 0.1 -> 0.01
epochs: 100 -> 10
-----
top1: 60.6982% -> 80.2928%
```

[homework-2/log/train_lr0.01_epoch10/20230606_171745/vis_data/20230606_171745.json](log/train_lr0.01_epoch10/20230606_171745/vis_data/20230606_171745.json)
