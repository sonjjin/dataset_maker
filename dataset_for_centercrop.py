import json
import os
import torch
import numpy as np
import cv2
import albumentations
import albumentations.pytorch
# from torchvision.io import read_image

class Radiate_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_folder, train_mode, transform = False):
        self.root_dir = data_folder
        self.folders = []
        self.transform = transform
        # print(len(os.listdir(self.root_dir)))
        self.train_mode = train_mode
        for curr_dir in os.listdir(self.root_dir):
            with open(os.path.join(self.root_dir, curr_dir, 'meta.json')) as f:
                meta = json.load(f)
            if meta["set"] == train_mode:
                self.folders.append(curr_dir)
        self.folders.sort()
        # print(self.folders)
        # print(len(self.folders))
        
        self.radar_dicts = self.get_radar_dicts(self.folders)
        # print(len(self.radar_dicts))
        # self.datas = []
        self.data_idx = 0        
        # for idx, _ in enumerate(self.radar_dicts):
        #     record_init = {}
        #     record_init["data_idx"] = self.data_idx
        #     record_init["img"] = self.radar_dicts[idx]
        #     record_init["train_mode"] = self.train_mode
        #     self.data_idx += 1
        #     self.datas.append(record_init)
        
    def __len__(self):
        return len(self.radar_dicts)
    
    def __getitem__(self, idx):
        radar_dict = self.radar_dicts[idx]
        img = cv2.imread(radar_dict["file_name"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # print(radar_dict["img"])
        boxes = []
        boxes_angle = []
        bbox_OOD = []
        category_id = []
        angle = []
        target = {}
        img_info = {}
        num_objs = 0
        difficulties = []

        for i in range(len(radar_dict['annotations'])):
            boxes.append(radar_dict['annotations'][i]['bbox'])
            boxes_angle.append(radar_dict['annotations'][i]['bbox_angle'])
            bbox_OOD.append(radar_dict['annotations'][i]['bbox_OOD'])
            category_id.append(radar_dict['annotations'][i]['category_id'])
            difficulties.append(['0'])
            num_objs += 1
        
        boxes = torch.as_tensor(boxes, dtype = torch.float32)
        angle = torch.as_tensor(angle, dtype = torch.float32)
        bbox_OOD = torch.as_tensor(bbox_OOD, dtype = torch.float32)
        category_id = torch.as_tensor(category_id, dtype = torch.float32)
        iscrowd = torch.zeros((num_objs,), dtype = torch.int64)
    
        target["file_name"] = radar_dict['file_name']
        target["frame_number"] = radar_dict['frame_number']
        target["folder_name"] = radar_dict['folder_name']
        target["bboxes"] = boxes
        target["bboxes_angle"] = boxes_angle
        target["bboxes_OOD"] = bbox_OOD
        target["category_id"] = category_id
        target['image_id'] = radar_dict['image_id']
        # target["iscrowd"] = iscrowd 
        target["difficulty"] = difficulties
        # labels = category_id
        img_info["folder_name"] = radar_dict['folder_name']
        img_info["frame_number"] = radar_dict['frame_number']
        if self.transform:
            data_transform = albumentations.Compose([
                albumentations.CenterCrop(256,256)],
                # albumentations.pytorch.transforms.ToTensorV2()],
                bbox_params=albumentations.BboxParams(format='pascal_voc', label_fields=['labels']),
                )
            data_transform_yolo = albumentations.Compose([
                albumentations.CenterCrop(256,256)],
                # albumentations.pytorch.transforms.ToTensorV2()],
                bbox_params=albumentations.BboxParams(format='yolo', label_fields=['labels']),
                )

            transformed = data_transform(image = img, bboxes = target['bboxes'], labels = target["category_id"])
            transformed_2 = data_transform_yolo(image = img, bboxes = target['bboxes_angle'], labels = target["category_id"])
            img = transformed['image']
            target['bboxes'] = torch.as_tensor(transformed['bboxes'], dtype=torch.float32)
            target['category_id'] = torch.as_tensor(transformed['labels'], dtype=torch.float32)
            target['bboxes_angle'] = torch.as_tensor(transformed_2['bboxes'], dtype=torch.float32)
            # labels = transformed['labels']
        
        return img, target, img_info
    


    def gen_boundingbox_OBB(self, bbox, angle):
        theta = np.deg2rad(-angle)
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
        points = np.array([[bbox[0], bbox[1]],
                           [bbox[0] + bbox[2], bbox[1]],
                           [bbox[0] + bbox[2], bbox[1] + bbox[3]],
                           [bbox[0], bbox[1] + bbox[3]]]).T

        cx = bbox[0] + bbox[2] / 2
        cy = bbox[1] + bbox[3] / 2
        T = np.array([[cx], [cy]])

        points = points - T
        points = np.matmul(R, points) + T
        points = points.astype(float)
        
        return points

    def gen_boundingbox_HBB(self, bbox, angle):
        theta = np.deg2rad(-angle)
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
        points = np.array([[bbox[0], bbox[1]],
                           [bbox[0] + bbox[2], bbox[1]],
                           [bbox[0] + bbox[2], bbox[1] + bbox[3]],
                           [bbox[0], bbox[1] + bbox[3]]]).T

        cx = bbox[0] + bbox[2] / 2
        cy = bbox[1] + bbox[3] / 2
        T = np.array([[cx], [cy]])

        points = points - T
        points = np.matmul(R, points) + T
        points = points.astype(float)

        min_x = np.min(points[0, :])
        min_y = np.min(points[1, :])
        max_x = np.max(points[0, :])
        max_y = np.max(points[1, :])

        return min_x, min_y, max_x, max_y
    
    def get_radar_dicts(self, folders):
        dataset_dicts = []
        idd = 0
        folder_size = len(folders)
        file_lenght = 0
        for idx, folder in enumerate(folders):
            radar_folder = os.path.join(self.root_dir, folder, 'Navtech_Cartesian')
            annotation_path = os.path.join(self.root_dir,
                                           folder, 'annotations', 'annotations.json')
            with open(annotation_path, 'r') as f_annotation:
                annotation = json.load(f_annotation)

            folder_number = idx
            radar_files = os.listdir(radar_folder)
            radar_files.sort()
            # print(radar_files)
            
            for frame_number in range(len(radar_files)):
                # if folder_number == 0:
                #     print(frame_number)
                record = {}
                objs = []
                bb_created = True
                idd += 1
                filename = os.path.join(
                    radar_folder, radar_files[frame_number])
                # print(filename)
                if (not os.path.isfile(filename)):
                    print(filename)
                    continue
                record["folder_name"] = folder
                record["frame_number"] = radar_files[frame_number]
                record["file_name"] = filename
                record["image_id"] = idd-1
                record["height"] = 1152
                record["width"] = 1152
                record["folder_number"] = folder_number

                for object in annotation:
                    if (object['bboxes'][frame_number]):
                        class_obj = object['class_name']
                        if (class_obj != 'pedestrian' and class_obj != 'group_of_pedestrians'):
                            bbox = object['bboxes'][frame_number]['position']
                            angle = object['bboxes'][frame_number]['rotation']
                            # bb_created = True
                            # if cfg.MODEL.PROPOSAL_GENERATOR.NAME == "RRPN":
                            min_x, min_y, max_x, max_y = self.gen_boundingbox_HBB(bbox, angle)

                            # print(bbox, angle.shape)
                            points = self.gen_boundingbox_OBB(bbox, angle)
                            x1 = points[0][0]
                            y1 = points[1][0]
                            x2 = points[0][1]
                            y2 = points[1][1]
                            x3 = points[0][2]
                            y3 = points[1][2]
                            x4 = points[0][3]
                            y4 = points[1][3]
                            
                            
                            cx = (bbox[0] + bbox[2] / 2) / 1152
                            cy = (bbox[1] + bbox[3] / 2) / 1152
                            wid = (bbox[2]) / 1152
                            hei = (bbox[3]) / 1152
                            
                            obj = {
                                "bbox": [min_x, min_y, max_x, max_y],
                                "bbox_angle": [cx, cy, wid, hei, angle],
                                "bbox_OOD": [x1, y1, x2, y2, x3, y3, x4, y4],
                                "category_id": 0,
                                "iscrowd": 0
                            }

                            objs.append(obj)
                # if bb_created:
                record["annotations"] = objs
                dataset_dicts.append(record)
        # print(dataset_dicts)
        return dataset_dicts
    
    def collate_fn(self, batch):
        num_images = len(batch)
        images = list()
        bboxes = torch.Tensor(num_images, 32, 4).zero_()
        category_id = torch.Tensor(num_images, 32).zero_()
        # bboxes = list()
        # category_id = list()
        
        for i in range(len(batch)):
            images.append(batch[i][0])
            # bboxes.append(batch[i][1])
            # category_id.append(batch[i][2])
            num_boxes = min(batch[i][1].size(0),32)
            if num_boxes == 0:
                continue
            bboxes[i, :num_boxes, :] = batch[i][1][:num_boxes, :]
            category_id[i, :num_boxes] = batch[i][2][:num_boxes]

        images = torch.stack(images, dim=0).contiguous()

        return images, bboxes, category_id