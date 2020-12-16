import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

import numpy as np
import json
import csv
import h5py
import random
import os
import os.path

import cv2
from PIL import Image


def video_to_tensor(pic):
    """Convert a ``numpy.ndarray`` to tensor.
    Converts a numpy.ndarray (T x H x W x C)
    to a torch.FloatTensor of shape (C x T x H x W)

    Args:
         pic (numpy.ndarray): Video to be converted to tensor.
    Returns:
         Tensor: Converted video.
    """
    return torch.from_numpy(pic.transpose([3, 0, 1, 2]))



def load_video_frames(video_dir, transforms):
    video = cv2.VideoCapture(video_dir)
    frames = []
    if not video.isOpened():
        raise FileNotFoundError('no video named %s found.' % video_dir)
    while video.isOpened():
        ret, frame = video.read()
        if ret:
            # h,w,c = frame.shape
            # frame = (frame[:,:,[2,1,0]]/255.)*2-1
            frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame = transforms(frame)
            frame = frame.unsqueeze(1)
            frames.append(frame)
        else:
            break
    return torch.cat(frames, 1)



class Salads50_without_label(Dataset):

    def __init__(self, root, transforms=None):

        self.video_files = os.listdir(root)
        self.transforms = transforms
        self.root = root

    def __getitem__(self, index):
        video_path = self.video_files[index]
        vid = video_path.split('.')[0]
        img_tensor = load_video_frames(os.path.join(self.root, video_path), self.transforms)
        # print(img_tensor.shape)
        # exit()
        # imgs = self.transforms(imgs)
        return img_tensor, vid
    def __len__(self):
        return len(self.video_files)
