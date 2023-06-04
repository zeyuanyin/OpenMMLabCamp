# Homework 1

[[Description]](https://github.com/open-mmlab/OpenMMLabCamp/issues/97)

[[Code Base]](https://github.com/TommyZihao/MMPose_Tutorials/tree/main/2023/0524)

[[Data]](https://drive.google.com/file/d/1zeOMs3i-1cRw6QZESp5mUwH0iozx0RZW/view?usp=drive_link)


## Environment Setup

1. Install MMdetection & MMpose
2. Move data and runing python files to the corresponding data folder


Code file organization:
```
code/
├── mmdetection/
|   ├── data/
|       ├── Ear210_Keypoint_Dataset_coco/
|       ├── rtmdet-tiny-ear.py
|   ├── ...
├── mmpose/
|   ├── data/
|       ├── Ear210_Keypoint_Dataset_coco/
|       ├── rtmpose-s-ear.py
|   ├── ...
```


## Task 1: Object Detection

Run the following command to train the model:
```
cd mmdetection
python tools/train.py data/rtmdet_tiny_ear.py
```
~1 hour on RTX 4090 GPU.

Training log and model weights will be saved at `work_dirs` folder.

Run the following command to test the model:
```
python tools/test.py data/rtmdet_tiny_ear.py \
        work_dirs/rtmdet_tiny_ear/epoch_200.pth
```

## Task 2: Pose Estimation

Run the following command to train the model:
```
cd mmpose
python tools/train.py data/rtmpose_s_ear.py
```
~3 hour on RTX 4090 GPU.

Run the following command to test the model:
```
python tools/test.py data/rtmpose_s_ear.py \
        work_dirs/rtmpose_s_ear/epoch_300.pth
```


