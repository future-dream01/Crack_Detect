#coding=utf-8
import xml.dom.minidom
import os
import random
import shutil

def is_image_file(filename):
    return any(filename.endswith(ext) for ext in ['.jpg', '.jpeg', '.png'])

def get_image_name(txt_name, image_folder):
    base_name = txt_name.split('.txt')[0]
    for ext in ['.jpg', '.jpeg', '.png']:
        image_name = base_name + ext
        if os.path.exists(os.path.join(image_folder, image_name)):
            return image_name
    return None

jpg_path = './images'
xml_path = './txts/'
train_xml = './train/Annotations'
val_xml = './val/Annotations'
train_jpg = './train/JPEGImages'
val_jpg = './val/JPEGImages'
os.makedirs(train_xml, exist_ok=True)
os.makedirs(val_xml, exist_ok=True)
os.makedirs(train_jpg, exist_ok=True)
os.makedirs(val_jpg, exist_ok=True)

datas = os.listdir(xml_path)
train_index = [i for i in range(0, len(datas))]
val_index = random.sample(train_index, int(0.1*len(datas)))

for index in val_index:
    train_index.remove(index)

for id in val_index:
    image_name = get_image_name(datas[id], jpg_path)
    if image_name:
        shutil.move(os.path.join(xml_path, datas[id]), val_xml)
        shutil.move(os.path.join(jpg_path, image_name), val_jpg)
    else:
        print(datas[id])

for id in train_index:
    image_name = get_image_name(datas[id], jpg_path)
    if image_name:
        shutil.move(os.path.join(xml_path, datas[id]), train_xml)
        shutil.move(os.path.join(jpg_path, image_name), train_jpg)
    else:
        print(datas[id])