import json
import os
import cv2

def is_image_file(filename):
    return any(filename.endswith(ext) for ext in ['.jpg', '.jpeg', '.png'])

def get_image_format(file_name, image_folder):
    base_name = file_name.split('.txt')[0]
    for ext in ['.jpg', '.jpeg', '.png']:
        image_name = base_name + ext
        if os.path.exists(os.path.join(image_folder, image_name)):
            return image_name
    return None

for name in ['train', 'val']:
    txt_newpath = f'./{name}/Annotations'
    jpg_newpath = f'./{name}/JPEGImages'

    categories = [{"id": 1, "name": "1", "supercategory": "1"}]
    images = []
    dataset = {}
    count = 0
    annotations = []
    num = 0
    for file in os.listdir(txt_newpath):
        print(f"INFO: {num}/{len(os.listdir(txt_newpath))}")
        with open(os.path.join(txt_newpath, file)) as fd:
            lines = fd.readlines()

        image_format = get_image_format(file, jpg_newpath)
        if not image_format:
            continue  # Skip if image file does not exist

        img = cv2.imread(os.path.join(jpg_newpath, image_format))
        width_scale, height_scale = img.shape[1], img.shape[0]
        images_info = {"file_name": image_format, "id": file.split('.txt')[0], "height": height_scale, "width": width_scale}
        images.append(images_info)

        for line in lines:
            line = line.strip()
            count += 1
            x, y, w, h = [float(i) for i in line.split(' ')[1:5]]
            annotations_info = {
                'image_id': file.split('.txt')[0],
                'bbox': [x, y, w, h],
                'category_id': int(line.split(' ')[0]),
                'area': 0,
                'iscrowd': 0,
                'segmentation': [],
                'person_id': 0
            }
            annotations.append(annotations_info)
        num += 1

    dataset = {'annotations': annotations, 'images': images, 'categories': categories}
    with open(f'./shale_{name}_2021.json', 'w', encoding='utf-8') as fd:
        json.dump(dataset, fd, ensure_ascii=False)


