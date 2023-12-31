# some utils codes for api.py, e.g., image loader, video loader
import os
import json
from collections import defaultdict
import torch
import cv2
from PIL import Image


class Video4Detector():
    def __init__(self, video_path):
        self.video_path = video_path
    
    def __len__(self):
        return self.total_frame_num

    def __iter__(self):
        # load video
        video = cv2.VideoCapture(self.video_path)
        # attributes
        self.total_frame_num = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.fps = int(video.get(cv2.CAP_PROP_FPS))
        self.video = video
        self.current_frame = 0
        return self

    def __next__(self):
        flag, frame = self.video.read()
        self.current_frame += 1

        assert flag == True and self.video.isOpened()
        assert self.current_frame == int(self.video.get(cv2.CAP_PROP_POS_FRAMES))

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        return frame, None, None

    def close(self):
        self.video.release()


class Images4Detector():
    def __init__(self, images_dir, gt_json=None, img_type='PIL', **kwargs):
        '''
        img_type: str, one of 'PIL', 'cv2', 'plt'
        '''
        self.pro_type = kwargs['pro_type'] if 'pro_type' in kwargs else None
        # images
        def is_img(s):
            return s.endswith('.jpg') or s.endswith('.png')

        self.img_names = []
        self.imgid2anns = defaultdict(list)
        for imdir, jspath in zip(images_dir, gt_json):
            self.img_names += [os.path.join(imdir, s) for s in sorted(os.listdir(imdir)) if is_img(s)]
            self.img_dir = imdir
            if self.pro_type == 'torch':
                self.imread = Image.open
            elif self.pro_type == 'cv2':
                self.imread = cv2.imread
            # ground truths
            if jspath:
                self.load_gt(jspath)
        self.total_frame_num = len(self.img_names)
        print('\nINFO: Test Datasets are {} ...'.format(self.total_frame_num))

    
    def load_gt(self, gt_json):
        with open(gt_json, 'r') as f:
            json_data = json.load(f)
        imgid2anns = defaultdict(list)
        for ann in json_data['annotations']:
            img_id = ann['image_id']
            ann['bbox'] = torch.Tensor(ann['bbox'])
            imgid2anns[img_id].append(ann)
        self.imgid2anns = imgid2anns

    def __len__(self):
        return self.total_frame_num
    
    def __iter__(self):
        self.i = -1
        return self
    
    def __next__(self):
        self.i += 1
        img_path = self.img_names[self.i]
        # load frame
        frame = self.imread(img_path)
        # load ground truth
        image_id = img_path.split('/')[-1][:-4]
        if self.imgid2anns:
            anns = self.imgid2anns[image_id]
        else:
            anns = None
        return frame, anns, image_id
