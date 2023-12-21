import os
import json
import random
from PIL import Image
import numpy as np
from collections import defaultdict

import torch
import torchvision.transforms.functional as tvf

from utils_scrpits.utils import normalize_bbox, rect_to_square
import utils_scrpits.augmentation as augUtils
import cv2
import copy

class Dataset4YoloAngle(torch.utils.data.Dataset):
    """
    dataset class.
    """
    def __init__(self, img_dir, json_path, img_size=(512, 512), augmentation=True, pro_type='torch'):
        """
        dataset initialization. Annotation data are read into memory by API.

        Args:
            img_dir: str or list, imgs folder, e.g. 'someDir/COCO/train2017/'
            json_path: str or list, e.g. 'someDir/COCO/instances_train2017.json'
            img_size: int, target image size input to the YOLO, default: 608
            augmentation: bool, default: True
            only_person: bool, if true, non-person BBs are discarded. default: True
            debug: bool, if True, only one data id is selected from the dataset
        """
        self.pro_type = pro_type
        self.max_labels = 50
        self.img_size = img_size
        self.enable_aug = augmentation
        self.img_ids = []
        self.imgid2info = dict()
        self.imgid2path = dict()
        self.imgid2anns = defaultdict(list)
        self.catids = []
        if isinstance(img_dir, str):
            assert isinstance(json_path, str)
            img_dir, json_path = [img_dir], [json_path]
        assert len(img_dir) == len(json_path)
        for imdir, jspath in zip(img_dir, json_path):
            self.load_anns(imdir, jspath)
        print('\nINFO: Training Datasets are {} ...'.format(len(self.img_ids)))
        random.shuffle(self.img_ids)

    def load_anns(self, img_dir, json_path):
        '''
        laod json file to self.img_ids, self.imgid2anns
        '''
        self.coco = False
        print('INFO: Loading annotations %s into memory...'%json_path)
        with open(json_path, 'r') as f:
            json_data = json.load(f)
        for ann in json_data['annotations']:
            img_id = ann['image_id']
            if len(ann['bbox'])==0:
                self.imgid2anns[img_id] = []
            else:
                ann['bbox'] = torch.Tensor(ann['bbox'])
                self.imgid2anns[img_id].append(ann)
        for img in json_data['images']:
            img_id = img['id']
            try:
                assert img_id not in self.imgid2path
            except Exception as e:
                print(img_id)
            anns = self.imgid2anns[img_id]
            # if there is crowd gt, skip this image
            if self.coco and any(ann['iscrowd'] for ann in anns):
                continue
            self.img_ids.append(img_id)
            self.imgid2path[img_id] = os.path.join(img_dir, img['file_name'])
            self.imgid2info[img['id']] = img
        # shuffle datasets
        random.shuffle(self.img_ids)
        self.catids = [cat['id'] for cat in json_data['categories']]

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index):
        """
        One image / label pair for the given index is picked up and pre-processed.

        Args:
        index (int): data index
        """
        # laod image
        img_id = self.img_ids[index]
        # img_name = self.imgid2info[img_id]['file_name']
        # img_path = os.path.join(self.img_dir, img_name)
        img_path = self.imgid2path[img_id]
        self.coco = True if 'COCO' in img_path else False
        if self.pro_type == 'torch':
            img = Image.open(img_path)
            ori_w, ori_h = img.width, img.height
        else:
            img = cv2.imread(img_path)

        # load unnormalized annotation
        annotations = self.imgid2anns[img_id]
        labels = torch.zeros(self.max_labels, 4)
        categories = torch.zeros(self.max_labels, dtype=torch.int64)
        li = 0
        for ann in annotations:
            if li >= 50:
                print(self.only_person)
                print(categories)
                break
            labels[li,:] = ann['bbox']
            categories[li] = self.catids.index(ann['category_id']+1)
            li += 1
        gt_num = li

        # pad to square
        input_img, labels[:gt_num], pad_info = rect_to_square(img, labels[:gt_num],
                                                              self.img_size, self.pro_type, pad_value=0,
                                                              aug=self.enable_aug)

        if self.pro_type=='torch':
            input_ori = tvf.to_tensor(input_img)
            img = input_ori
            # img = input_ori.unsqueeze(0)
        else:
            input_ori = torch.from_numpy(input_img).permute((2, 0, 1)).float() / 255.
            img = input_ori
            # img = input_ori.unsqueeze(0)
        if self.enable_aug:
            # if np.random.rand() > 0.5:
            #     img = augUtils.add_gaussian(img, max_var=0.03)
            blur = [augUtils.random_avg_filter, augUtils.max_filter,
                    augUtils.random_gaussian_filter]
            if not self.coco and np.random.rand() > 0.8:
                blur_func = random.choice(blur)
                img = blur_func(img)
            # if np.random.rand() > 0.5:
            #     img = augUtils.add_saltpepper(img, max_p=0.04)

        labels[:gt_num] = normalize_bbox(labels[:gt_num], self.img_size[1], self.img_size[0])

        return img, labels, categories, str(img_id), pad_info
