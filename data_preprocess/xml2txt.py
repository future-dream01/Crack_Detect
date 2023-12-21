# -*- coding:utf-8 -*-
import cv2
import os
import shutil
from xml.etree.ElementTree import ElementTree

def is_image_file(filename):
    return any(filename.endswith(ext) for ext in ['.jpg', '.jpeg', '.png'])

def get_image_format(base_path, xml_name):
    for img_format in ['.jpg', '.jpeg', '.png']:
        img_name = base_path + xml_name.replace('.xml', img_format)
        if os.path.exists(img_name):
            return img_format
    return None

xml_path = './xml/'
base_path = './images/'
save_img = "./save-imgs/"
os.makedirs(save_img, exist_ok=True)
xml_list = os.listdir(xml_path)
error_data = './error_data/'
os.makedirs(error_data, exist_ok=True)
save_txt = "./txts/"
os.makedirs(save_txt, exist_ok=True)

for name in xml_list:
    img_format = get_image_format(base_path, name)
    if not img_format:
        continue  # 如果找不到对应的图片格式，则跳过

    txt_name = name.replace('xml', 'txt')
    with open(os.path.join(save_txt, txt_name), 'w') as txt_file:
        tree = ElementTree()
        tree.parse(os.path.join(xml_path, name))
        targets = tree.findall('./object/bndbox')
        out_box = []

        for bb in targets:
            xmin, ymin, xmax, ymax = [float(bb.find(corner).text) for corner in ['xmin', 'ymin', 'xmax', 'ymax']]
            cx, cy, w, h = (xmin + xmax) / 2, (ymin + ymax) / 2, xmax - xmin, ymax - ymin
            txt_file.write(f'0 {cx} {cy} {w} {h}\n')

        if targets:
            img_name = base_path + name.replace('.xml', img_format)
            if os.path.exists(img_name):
                img = cv2.imread(img_name)
                for bb in targets:
                    xmin, ymin, xmax, ymax = [int(float(bb.find(corner).text)) for corner in ['xmin', 'ymin', 'xmax', 'ymax']]
                    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
                cv2.imwrite(os.path.join(save_img, name.replace('.xml', img_format)), img)