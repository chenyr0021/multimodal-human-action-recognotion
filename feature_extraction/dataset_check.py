import cv2 as cv
import os

def dataset_check(root):
    img_dirs = os.listdir(root)
    for img_dir in img_dirs:
        img_paths = os.listdir(os.path.join(root, img_dir))
        print(img_dir)
        for img_path in img_paths:
            try:
                img = cv.imread(os.path.join(root, img_dir, img_path))
            except:
                print('Read image error. Path: ', img_path)
                continue

if __name__ == '__main__':
    # dataset_check('/home/dataset/Charades_v1_rgb')
    root = '/home/dataset/Charades_v1_rgb/W97NR'
    paths = os.listdir(root)
    for p in paths:
        print(p)
        img = cv.imread(os.path.join(root, p))
