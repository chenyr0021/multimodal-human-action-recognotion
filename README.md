# multimodal-human-action-recognotion


《基于深度学习的多模态人体行为识别》论文工作整理及总结。


![image](https://github.com/chenyr0021/multimodal-human-action-recognotion/blob/main/pictures/%E5%B7%A5%E4%BD%9C%E6%80%BB%E7%BB%93.png)

This work is mainly composed of 3 parts:

* [Data collection](#data-collection)
* [Feature extraction](#feature-extraction)
* [Classification](#classification)

# Tested environment

Ubuntu 16.04
Pytorch 
CUDA

# Data collection

We use 2 different modalities in this work, which are video and Inertial data. The sensors we used are USB camera and IMU([MetaMotionR from MbientLab](https://mbientlab.com/store/)), respectively.

![image](https://github.com/chenyr0021/multimodal-human-action-recognotion/blob/main/pictures/sensors.png)

The IMU device should be used with its own package. See tutorial in https://mbientlab.com/tutorials/CppDevelopment.html to install it.

## Usage

Code for data collection is in folder `dataset collector`. To collect data, plug in USB camera and run:

```
python3 generate_data.py
```

NOTE: The IMU device id can be changed in this file.

Running this code will generate 2 .csv files ending with '0' or '1' for 2 IMU devices respectively, another .csv file for annotation, and an .mp4 file for camera. 

E.g.

- data
  - 20201216-xxxxxx.csv (annotation file)
  - 20201216-xxxxxx_0.csv (left IMU)
  - 20201216-xxxxxx_1.csv (right IMU)
  - 20201216-xxxxxx.mp4 (video)
  
Here, xxxxxx is timestamp. 

The 4th column of IMU file is the video sequence number which the corresponding IMU frame should be synchronized with.

When labelling data, you can fill the 3rd column in annotation file with class number of each row.

`divide_video_frame.py` is to divide video file into individual frames.

`./data/label_file_transform.py` is used to generate groundtruth file from annotation to match classification code.

`util.py` is used to divide train/test set.



# Feature extraction

In this work, we extract feature from intertial data and video independently.

`extract_features.py` is used to extract video feature. This algorithm is i3d: https://github.com/piergiaj/pytorch-i3d.

Code for inertial feature extraction is in folder `IMU_data_process`.



# Classification

See https://github.com/chenyr0021/multimodal-human-action-recognotion/tree/main/classification
