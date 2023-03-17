import json
import os
import torch
import numpy as np
import cv2
import albumentations
import albumentations.pytorch
import pdb
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

def main():

    root_dir = '/workspace/dataset/radiate_512'
    dataset = Radiate_Dataset(data_folder, train_mode, transform=True)

    img_radar_save = os.path.join(root_dir, train_mode, 'images_radar')
    img_lidar_save = os.path.join(root_dir, train_mode, 'images_lidar')
    # img_add_save = os.path.join(root_dir, train_mode, 'images_add')
    
    lidar_save = os.path.join(root_dir, train_mode, 'lidar_info')
    
    label_save = os.path.join(root_dir, train_mode, 'labels')
    label_angle_save = os.path.join(root_dir, train_mode, 'labels_angle')
    label_OBB_save = os.path.join(root_dir, train_mode, 'labelTxt')
    img_info_save = os.path.join(root_dir, train_mode, 'img_info')
    os.makedirs(img_radar_save, exist_ok=True)
    os.makedirs(img_lidar_save, exist_ok=True)
    # os.makedirs(img_add_save, exist_ok=True)
    os.makedirs(label_save, exist_ok=True)
    os.makedirs(label_angle_save, exist_ok=True)
    os.makedirs(label_OBB_save, exist_ok=True)
    os.makedirs(img_info_save, exist_ok=True)
    os.makedirs(lidar_save, exist_ok=True)

    for i in tqdm(range(len(dataset))):
        img_radar, img_lidar, target, img_info = dataset[i]
        # img_add = img_radar+img_lidar
        file_name = target['file_name']
        bbox = target['bboxes']
        bbox_angle = target['bboxes_angle']
        category = target['category_id']
        img_radar_save_path = os.path.join(img_radar_save, str(i).zfill(5)+'.png')
        img_lidar_save_path = os.path.join(img_lidar_save, str(i).zfill(5)+'.png')
        # img_add_save_path = os.path.join(img_add_save, str(i).zfill(5)+'.png')
        lidar_info_save_path = os.path.join(lidar_save, str(i).zfill(5)+'.csv')
        label_save_path = os.path.join(label_save, str(i).zfill(5)+'.txt')
        label_angle_save_path = os.path.join(label_angle_save, str(i).zfill(5)+'.txt')
        label_OBB_save_path = os.path.join(label_OBB_save, str(i).zfill(5)+'.txt')
        img_info_save_path = os.path.join(img_info_save, str(i).zfill(5)+'.json')
        
        # print(bbox_angle)
        # print(img_save_path)
        cv2.imwrite(img_radar_save_path, img_radar)
        cv2.imwrite(img_lidar_save_path, img_lidar)
        shutil.copy2(img_info['lidar_path'], lidar_info_save_path)
        # cv2.imwrite(img_add_save_path, img_add)
        
        # save HBB (cx, cy, wid, hei)
        with open(label_save_path, 'w') as f:
            for j in range(bbox.shape[0]):
                # print("%i %.6f %.6f %.6f %.6f" % (category[j], bbox[j][0], bbox[j][1], bbox[j][2], bbox[j][3]))
                # print(str(np.array(category[j]).astype(np.int32)))
                cx = (bbox[j][0] + bbox[j][2]) / 2
                cy = (bbox[j][1] + bbox[j][3]) / 2
                wid = bbox[j][2] - bbox[j][0]
                hei = bbox[j][3] - bbox[j][1]
                f.write("%i %.6f %.6f %.6f %.6f \n" % (category[j], cx, cy, wid, hei))
                
        # save OBB(cx, cy, wid, hei, angle)
        with open(label_angle_save_path,"w") as f:
            for j in range(bbox_angle.shape[0]):
                # print("%i %.6f %.6f %.6f %.6f" % (category[j], bbox[j][0], bbox[j][1], bbox[j][2], bbox[j][3]))
                # print(str(np.array(category[j]).astype(np.int32)))
                cx = bbox_angle[j][0]*img_radar.shape[0]
                cy = bbox_angle[j][1]*img_radar.shape[0]
                wid = bbox_angle[j][2]*img_radar.shape[0]
                hei = bbox_angle[j][3]*img_radar.shape[0]
                angle = bbox_angle[j][4]
                f.write("%.6f %.6f %.6f %.6f %.6f vehicle %i \n" % (cx, cy, wid, hei, angle, 0))
        
        with open(label_OBB_save_path,"w") as f:
            bbox_OBB = []
            for k, HBB in enumerate(bbox_angle):
                points = gen_boundingbox(HBB*img_radar.shape[0], HBB[4])
                x1 = points[0][0]
                y1 = points[1][0]
                x2 = points[0][1]
                y2 = points[1][1]
                x3 = points[0][2]
                y3 = points[1][2]
                x4 = points[0][3]
                y4 = points[1][3]
                if x1 < 0:
                    x1 = 0
                if x2 < 0:
                    x2 = 0
                if x3 < 0:
                    x3 = 0
                if x4 < 0:
                    x4 = 0
                if y1 < 0:
                    y1 = 0
                if y2 < 0:
                    y2 = 0
                if y3 < 0:
                    y3 = 0
                if y4 < 0:
                    y4 = 0
                bbox_OBB.append([x1, y1, x2, y2, x3, y3, x4, y4])
            for j in range(bbox_angle.shape[0]):
                # print("%i %.6f %.6f %.6f %.6f" % (category[j], bbox[j][0], bbox[j][1], bbox[j][2], bbox[j][3]))
                # print(str(np.array(category[j]).astype(np.int32)))
                f.write("%.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f vehicle %i \n" % (bbox_OBB[j][0], bbox_OBB[j][1], bbox_OBB[j][2], bbox_OBB[j][3], bbox_OBB[j][4], bbox_OBB[j][5], bbox_OBB[j][6], bbox_OBB[j][7], 0))
        
        with open(img_info_save_path, 'w', encoding='utf-8') as f:
            json.dump(img_info, f)

        
    
    
if __name__ == '__main__':
    main()