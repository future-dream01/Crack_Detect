import os
import time
import argparse
from utils_scrpits.parse_config import *
from utils_scrpits.utils import *
from utils_scrpits.get_module_list import *
import cv2
import torch
from PIL import Image

parser = argparse.ArgumentParser(description="demo for rotated image detection")
parser.add_argument('--model_def', type=str, default='cfg/model.cfg')
parser.add_argument('--data_config', type=str, default='cfg/cls.names')
parser.add_argument('--model', type=str, default='weights/CrackDetection_Dec21-23_160.pth', help='model path')
parser.add_argument('--img_path', type=str, default='val/JPEGImages', help='image path')
parser.add_argument('--input_size', type=int, default=(640, 640))
parser.add_argument("--conf_thres", type=float, default=0.5, help="object confidence threshold")
parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
parser.add_argument('--visualize', type=bool, default=True)
parser.add_argument('--preprocess_type', type=str, default='cv2', choices=['cv2', 'torch'], help='image preprocess type')
args = parser.parse_args()

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
module_defs = parse_model_config(args.model_def)
model = RotateDetectNet(module_defs)
pretrained_dict = torch.load(args.model)
model.load_state_dict(pretrained_dict['model'])
classes = parse_data_config(args.data_config)['names'].split(',')
model.to(device)

# Create result directory if it doesn't exist
result_path = "result/"
os.makedirs(result_path, exist_ok=True)

with torch.no_grad():
    for root, folder, files in os.walk(args.img_path):
        if len(files) > 0:
            count_num = 0
            print("INFO: " + root)
            for name in files:
                # Check if the file is an image
                if not (name.endswith('.jpg') or name.endswith('.jpeg') or name.endswith('.png')):
                    continue
                count_num += 1
                print('INFO: %d/%d' % (count_num, len(files)))
                pro_type = args.preprocess_type

                # Load and preprocess image
                start = time.time()
                if pro_type == 'torch':
                    pil_img = Image.open(os.path.join(root, name))
                else:
                    pil_img = cv2.imread(os.path.join(root, name))

                input_img, _, pad_info = rect_to_square(pil_img, None, args.input_size, pro_type, 0)
                input_ori = torch.from_numpy(input_img).permute((2, 0, 1)).float() / 255.
                input_ = input_ori.unsqueeze(0)

                # Perform detection
                input_ = input_.cuda()
                dts = model(input_).cpu()
                np_img = np.array(pil_img)

                dts=dts.view(-1,5)
                mask=dts[:,4]>=args.conf_thres
                dts=dts[mask]

                # Post-processing
                #dts = dts[dts[:, 4:].max(-1)[0] >= args.conf_thres]
                if len(dts):
                    detections = non_max_suppression(dts, args.conf_thres, args.nms_thres)
                    detections = detection2original(detections, pad_info.squeeze())
                    for bb in detections:
                        cv2.putText(np_img, classes[int(bb[-1])] + ': %.2f' % (bb[-2]), (int(bb[0]), int(bb[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                        cv2.rectangle(np_img, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), color=(255, 0, 0), thickness=1, lineType=cv2.LINE_4)

                # Save and display result
                cv2.imwrite(os.path.join(result_path, name), np_img)
                if args.visualize:
                    cv2.namedWindow('result', cv2.WINDOW_NORMAL)
                    cv2.imshow('result', np_img)
                    cv2.waitKey()

print("Detection completed.")