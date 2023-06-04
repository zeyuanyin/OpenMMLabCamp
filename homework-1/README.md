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
python tools/test.py data/rtmpose-s-ear.py \
        work_dirs/rtmpose-s-ear/epoch_300.pth
```


## Results

Object Detection (test on epoch_200.pth)

[homework-1/log/mmdetection_test/20230604_185806.json](log/mmdetection_test/20230604_185806.json)

```
{"coco/bbox_mAP": 0.792, "coco/bbox_mAP_50": 0.955, "coco/bbox_mAP_75": 0.946, "coco/bbox_mAP_s": -1.0, "coco/bbox_mAP_m": -1.0, "coco/bbox_mAP_l": 0.792, "data_time": 0.2294788360595703, "time": 0.34926186908375134}
```

Pose Estimation (test on epoch_190.pth)

[homework-1/log/mmpose_test_E190/20230604_195520.json](log/mmdetection_test/20230604_185806.json)
```
{"coco/AP": 0.6579857911246617, "coco/AP .5": 1.0, "coco/AP .75": 0.7857978775631519, "coco/AP (M)": -1.0, "coco/AP (L)": 0.6579857911246617, "coco/AR": 0.7142857142857143, "coco/AR .5": 1.0, "coco/AR .75": 0.8571428571428571, "coco/AR (M)": -1.0, "coco/AR (L)": 0.7142857142857143, "PCK": 0.9523809523809522, "AUC": 0.09971655328798187, "NME": 0.04699088548586752, "data_time": 1.4403223594029744, "time": 1.6241701046625774}
```