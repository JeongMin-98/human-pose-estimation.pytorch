# Simple Baselines for Hand/Foot Keypoint Detection (Xray Image)
forked Simple Baseline [Microsoft](https://github.com/microsoft/human-pose-estimation.pytorch)

## Introduction
This is Hand/Foot Keypoint Detection code that forked an official pytorch implementation of [*Simple Baselines for Human Pose Estimation and Tracking*](https://arxiv.org/abs/1804.06208). 
I implement all models for research purpose

## Main Results
I will record main results from my experiments
[//]: # (### Note:)

[//]: # (- Flip test is used.)
[//]: # ()
[//]: # (### Results on COCO val2017 with detector having human AP of 56.4 on COCO val2017 dataset)

[//]: # (| Arch | AP | Ap .5 | AP .75 | AP &#40;M&#41; | AP &#40;L&#41; | AR | AR .5 | AR .75 | AR &#40;M&#41; | AR &#40;L&#41; |)

[//]: # (|---|---|---|---|---|---|---|---|---|---|---|)

[//]: # (| 256x192_pose_resnet_50_d256d256d256 | 0.704 | 0.886 | 0.783 | 0.671 | 0.772 | 0.763 | 0.929 | 0.834 | 0.721 | 0.824 |)

[//]: # (| 384x288_pose_resnet_50_d256d256d256 | 0.722 | 0.893 | 0.789 | 0.681 | 0.797 | 0.776 | 0.932 | 0.838 | 0.728 | 0.846 |)

[//]: # (| 256x192_pose_resnet_101_d256d256d256 | 0.714 | 0.893 | 0.793 | 0.681 | 0.781 | 0.771 | 0.934 | 0.840 | 0.730 | 0.832 |)

[//]: # (| 384x288_pose_resnet_101_d256d256d256 | 0.736 | 0.896 | 0.803 | 0.699 | 0.811 | 0.791 | 0.936 | 0.851 | 0.745 | 0.858 |)

[//]: # (| 256x192_pose_resnet_152_d256d256d256 | 0.720 | 0.893 | 0.798 | 0.687 | 0.789 | 0.778 | 0.934 | 0.846 | 0.736 | 0.839 |)

[//]: # (| 384x288_pose_resnet_152_d256d256d256 | 0.743 | 0.896 | 0.811 | 0.705 | 0.816 | 0.797 | 0.937 | 0.858 | 0.751 | 0.863 |)

[//]: # ()
[//]: # ()
[//]: # (### Results on *Caffe-style* ResNet)

[//]: # (| Arch | AP | Ap .5 | AP .75 | AP &#40;M&#41; | AP &#40;L&#41; | AR | AR .5 | AR .75 | AR &#40;M&#41; | AR &#40;L&#41; |)

[//]: # (|---|---|---|---|---|---|---|---|---|---|---|)

[//]: # (| 256x192_pose_resnet_50_*caffe*_d256d256d256 | 0.704 | 0.914 | 0.782 | 0.677 | 0.744 | 0.735 | 0.921 | 0.805 | 0.704 | 0.783 |)

[//]: # (| 256x192_pose_resnet_101_*caffe*_d256d256d256 | 0.720 | 0.915 | 0.803 | 0.693 | 0.764 | 0.753 | 0.928 | 0.821 | 0.720 | 0.802 |)

[//]: # (| 256x192_pose_resnet_152_*caffe*_d256d256d256 | 0.728 | 0.925 | 0.804 | 0.702 | 0.766 | 0.760 | 0.931 | 0.828 | 0.729 | 0.806 |)


[//]: # (### Note:)

[//]: # (- Flip test is used.)

[//]: # (- Person detector has person AP of 56.4 on COCO val2017 dataset.)

[//]: # (- Difference between *PyTorch-style* and *Caffe-style* ResNet is the position of stride=2 convolution)

## Environment
The original code is developed using python 3.6 on Ubuntu 16.04 and developed and tested using 4 NVIDIA P100 GPU cards. 
Other platforms or GPU cards are not fully tested. thus I fix code for my environment. 
My code is developed using python 3.8 on Ubuntu 18.04+(WSL) and tested using 1 NVIDIA GeForce RTX 2070 GPU cards. 
My code is also tested using 1 NVIDIA GeForce RTX 3060 GPU cards.

## Quick start
### Installation
MayBe You could not follow below step(1~3)
1. Install pytorch >= v0.4.0 following [official instruction](https://pytorch.org/).
2. Disable cudnn for batch_norm:
   ```
   # PYTORCH=/path/to/pytorch
   # for pytorch v0.4.0
   sed -i "1194s/torch\.backends\.cudnn\.enabled/False/g" ${PYTORCH}/torch/nn/functional.py
   # for pytorch v0.4.1
   sed -i "1254s/torch\.backends\.cudnn\.enabled/False/g" ${PYTORCH}/torch/nn/functional.py
   ```
   Note that instructions like # PYTORCH=/path/to/pytorch indicate that you should pick a path where you'd like to have pytorch installed  and then set an environment variable (PYTORCH in this case) accordingly.
1. Clone this repo, and we'll call the directory that you cloned as ${POSE_ROOT}.

### Dependencies
+ Anaconda (recommend)
   ```
   conda env create -f environment.yml
   ```
1. Make libs:
   ```
   cd ${POSE_ROOT}/lib
   make
   ```
2. Install [COCOAPI](https://github.com/cocodataset/cocoapi):
   ```
   # COCOAPI=/path/to/clone/cocoapi
   git clone https://github.com/cocodataset/cocoapi.git $COCOAPI
   cd $COCOAPI/PythonAPI
   # Install into global site-packages
   make install
   # Alternatively, if you do not have permissions or prefer
   # not to install the COCO API into global site-packages
   python3 setup.py install --user
   ```
   Note that instructions like # COCOAPI=/path/to/install/cocoapi indicate that you should pick a path where you'd like to have the software cloned and then set an environment variable (COCOAPI in this case) accordingly.
3. Download pytorch imagenet pretrained models from [pytorch model zoo](https://pytorch.org/docs/stable/model_zoo.html#module-torch.utils.model_zoo) and caffe-style pretrained models from [GoogleDrive](https://drive.google.com/drive/folders/1yJMSFOnmzwhA4YYQS71Uy7X1Kl_xq9fN?usp=sharing). 
4. Download mpii and coco pretrained models from [OneDrive](https://1drv.ms/f/s!AhIXJn_J-blW0D5ZE4ArK9wk_fvw) or [GoogleDrive](https://drive.google.com/drive/folders/13_wJ6nC7my1KKouMkQMqyr9r1ZnLnukP?usp=sharing). Please download them under ${POSE_ROOT}/models/pytorch, and make them look like this:

   ```
   ${POSE_ROOT}
    `-- models
        `-- pytorch
            |-- imagenet
            |   |-- resnet50-19c8e357.pth
            |   |-- resnet50-caffe.pth.tar
            |   |-- resnet101-5d3b4d8f.pth
            |   |-- resnet101-caffe.pth.tar
            |   |-- resnet152-b121ed2d.pth
            |   `-- resnet152-caffe.pth.tar
            |-- pose_coco
            |   |-- pose_resnet_101_256x192.pth.tar
            |   |-- pose_resnet_101_384x288.pth.tar
            |   |-- pose_resnet_152_256x192.pth.tar
            |   |-- pose_resnet_152_384x288.pth.tar
            |   |-- pose_resnet_50_256x192.pth.tar
            |   `-- pose_resnet_50_384x288.pth.tar
            `-- pose_mpii
                |-- pose_resnet_101_256x256.pth.tar
                |-- pose_resnet_101_384x384.pth.tar
                |-- pose_resnet_152_256x256.pth.tar
                |-- pose_resnet_152_384x384.pth.tar
                |-- pose_resnet_50_256x256.pth.tar
                `-- pose_resnet_50_384x384.pth.tar

   ```

4. Init output(training model output directory) and log(tensorboard log directory) directory:

   ```
   mkdir output 
   mkdir log
   ```

   Your directory tree should look like this:

   ```
   ${POSE_ROOT}
   ├── data
   ├── experiments
   ├── lib
   ├── log
   ├── models
   ├── output
   ├── pose_estimation
   ├── README.md
   └── requirements.txt
   ```
   
### Data preparation
**For MPII data**, please download from [MPII Human Pose Dataset](http://human-pose.mpi-inf.mpg.de/). The original annotation files are in matlab format. We have converted them into json format, you also need to download them from [OneDrive](https://1drv.ms/f/s!AhIXJn_J-blW00SqrairNetmeVu4) or [GoogleDrive](https://drive.google.com/drive/folders/1En_VqmStnsXMdldXA6qpqEyDQulnmS3a?usp=sharing).
Extract them under {POSE_ROOT}/data, and make them look like this:
```
${POSE_ROOT}
|-- data
`-- |-- mpii
    `-- |-- annot
        |   |-- gt_valid.mat
        |   |-- test.json
        |   |-- train.json
        |   |-- trainval.json
        |   `-- valid.json
        `-- images
            |-- 000001163.jpg
            |-- 000003072.jpg
```

**For COCO data**, please download from [COCO download](http://cocodataset.org/#download), 2017 Train/Val is needed for COCO keypoints training and validation. We also provide person detection result of COCO val2017 to reproduce our multi-person pose estimation results. Please download from [OneDrive](https://1drv.ms/f/s!AhIXJn_J-blWzzDXoz5BeFl8sWM-) or [GoogleDrive](https://drive.google.com/drive/folders/1fRUDNUDxe9fjqcRZ2bnF_TKMlO0nB_dk?usp=sharing).
Download and extract them under {POSE_ROOT}/data, and make them look like this:
```
${POSE_ROOT}
|-- data
`-- |-- coco
    `-- |-- annotations
        |   |-- person_keypoints_train2017.json
        |   `-- person_keypoints_val2017.json
        |-- person_detection_results
        |   |-- COCO_val2017_detections_AP_H_56_person.json
        `-- images
            |-- train2017
            |   |-- 000000000009.jpg
            |   |-- 000000000025.jpg
            |   |-- 000000000030.jpg
            |   |-- ... 
            `-- val2017
                |-- 000000000139.jpg
                |-- 000000000285.jpg
                |-- 000000000632.jpg
                |-- ... 
```

### Valid on MPII using pretrained models

```
python pose_estimation/valid.py \
    --cfg experiments/mpii/resnet50/256x256_d256x3_adam_lr1e-3.yaml \
    --flip-test \
    --model-file models/pytorch/pose_mpii/pose_resnet_50_256x256.pth.tar
```

### Training on MPII

```
python pose_estimation/train.py \
    --cfg experiments/mpii/resnet50/256x256_d256x3_adam_lr1e-3.yaml
```

### Valid on COCO val2017 using pretrained models

```
python pose_estimation/valid.py \
    --cfg experiments/coco/resnet50/256x192_d256x3_adam_lr1e-3.yaml \
    --flip-test \
    --model-file models/pytorch/pose_coco/pose_resnet_50_256x192.pth.tar
```

### Training on COCO train2017

```
python pose_estimation/train.py \
    --cfg experiments/coco/resnet50/256x192_d256x3_adam_lr1e-3.yaml
```

### Citation
If you use our code or models in your research, please cite with:
```
@inproceedings{xiao2018simple,
    author={Xiao, Bin and Wu, Haiping and Wei, Yichen},
    title={Simple Baselines for Human Pose Estimation and Tracking},
    booktitle = {European Conference on Computer Vision (ECCV)},
    year = {2018}
}
```
