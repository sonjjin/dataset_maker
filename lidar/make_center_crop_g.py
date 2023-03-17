import json
import os
import torch
import numpy as np
import cv2
import albumentations
import albumentations.pytorch
# from torchvision.io import read_image

import argparse
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataset_for_centercrop_lidar import Radiate_Dataset

import PIL
import numpy as np
import cv2
import shutil
from tqdm import tqdm
import pdb
parser = argparse.ArgumentParser()
parser.add_argument("--data_folder", help="root folder with radiate dataset", 
                    default='/workspace/dataset/radiate', type=str)

parser.add_argument("--save_folder", help="root for save directory",
                    default = "/workspace/JS/faster_rcnn_JS/checkpoint", type=str)

parser.add_argument("--train_mode", help="dataset mode ('train_good_weather', 'train_good_and_bad_weather', 'test')",
                    default='train_good_weather', type=str)




args = parser.parse_args()
data_folder = args.data_folder
train_mode = args.train_mode

def gen_boundingbox(bbox, angle):
    theta = np.deg2rad(-angle)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]])
    
    x1 = bbox[0] - bbox[2] / 2
    x2 = bbox[0] + bbox[2] / 2
    y1 = bbox[1] - bbox[3] / 2
    y2 = bbox[1] + bbox[3] / 2
    
    points = np.array([[x1, y1],
                       [x2, y1],
                       [x2, y2],
                       [x1, y2]]).T

    cx = bbox[0]
    cy = bbox[1]

    T = np.array([[cx], [cy]])

    points = points - T
    points = np.matmul(R, points) + T
    points = points.astype(float)
    
    return points

def centercrop(img, set_size=(512,512)):
    try:
        h,w,_ = img.shape
    except:
        h,w = img.shape
    crop_width = set_size[0]
    crop_height = set_size[1]
    
    mid_x, mid_y = w//2, h//2
    offset_x, offset_y = crop_width//2, crop_height//2
    img_out = img[mid_y - offset_y:mid_y+offset_y, mid_x - offset_x:mid_x+offset_x]
    return img_out

def main():

    root_dir = '/workspace/dataset/radiate_512'
    # dataset = Radiate_Dataset(data_folder, train_mode, transform=True)

    mask_save_path = os.path.join(root_dir, train_mode, 'mask')
    mask_origin_path = os.path.join('/workspace/dataset/radiate_origin', train_mode, 'mask')
    # img_add_save = os.path.join(root_dir, train_mode, 'images_add')
    img_lists = os.listdir(mask_origin_path)
    img_lists.sort()
    os.makedirs(mask_save_path, exist_ok=True)
    

    for img_list in tqdm(img_lists):
        

        mask_save_img = os.path.join(mask_save_path, img_list)
        mask_origin_img = os.path.join(mask_origin_path, img_list)
        # img_add_save_path = os.path.join(img_add_save, str(i).zfill(5)+'.png')
        
        mask_oirigin = cv2.imread(mask_origin_img)
        img_mask = centercrop(mask_oirigin)
        cv2.imwrite(mask_save_img, img_mask)
        # print(bbox_angle)
        # print(img_save_path)
       
    
    
if __name__ == '__main__':
    main()