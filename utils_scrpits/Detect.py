import numpy as np
from tqdm import tqdm
import torch
import torchvision.transforms.functional as tvf

from utils_scrpits import dataloader, utils
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline


class Detector():
    '''
    Wrapper of image object detectors.

    Args:
        model_name: str, currently only support 'rapid'
        weights_path: str, path to the pre-trained network weights
        model: torch.nn.Module, used only during training
        conf_thres: float, confidence threshold
        input_size: int, input resolution
    '''

    def __init__(self, model=None, **kwargs):
        assert torch.cuda.is_available()
        if model:
            self.model = model
            self.conf_thres = kwargs.get('conf_thres', None)
            self.nms_thres = kwargs.get('nms_thres', None)
            self.class_name = kwargs.get('class_name', None)
            self.input_size = kwargs.get('input_size', None)

            self.iou_thres = (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)
            self.rec_thres = torch.linspace(0, 1, steps=101)

    def forward(self, img_dir, **kwargs):
        '''
        Run on a sequence of images in a folder.

        Args:
            img_dir: str
            input_size: int, input resolution
            conf_thres: float, confidence threshold
        '''
        gt_path = kwargs['gt_path'] if 'gt_path' in kwargs else None

        ims = dataloader.Images4Detector(img_dir, gt_path, **kwargs)  # TODO
        dts = self._detect_iter(iter(ims), **kwargs)
        return dts

    def _detect_iter(self, iterator, **kwargs):
        Precision_iou_thres = np.zeros((len(self.class_name), len(self.iou_thres)))
        Recall_iou_thres = np.zeros((len(self.class_name), len(self.iou_thres)))
        AP_iou_thres = np.zeros((len(self.class_name), len(self.iou_thres)))
        labels = []
        tps = []
        for _ in tqdm(range(len(iterator))):
            sample_metrics = []  # List of tuples (TP, confs, pred)
            pil_frame, anns, img_id = next(iterator)
            if anns is None:
                continue
            # Extract labels
            targets = torch.empty((len(anns), 5))
            for ind, ann in enumerate(anns):
                labels += torch.Tensor([ann['category_id']]).tolist()
                # Transform target xywh2xyxy
                targets[ind, :] = torch.cat((torch.Tensor([ann['category_id']]), utils.xywh2xyxy(ann['bbox'])), 0)
            detections = self._predict_pil(pil_img=pil_frame, **kwargs)

            for tidx, iou_thres in enumerate(self.iou_thres):
                sample_metrics += utils.get_batch_statistics(detections, targets, iou_thres)
            if len(sample_metrics) != 0:
                tps.append(np.array(sample_metrics))

        # Concatenate sample statistics
        pre = []
        rec = []
        for tidx in range(len(self.iou_thres)):
            tp = np.array(tps)[:, tidx].tolist()
            if len(tp) == 0:  # no detections over whole validation set.
                continue
            true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*tp))]

            p, r, ap, ap_class, pre, rec = utils.ap_per_class(true_positives, pred_scores, pred_labels, labels,
                                                    self.rec_thres, self.iou_thres[tidx], pre, rec)
            Precision_iou_thres[..., tidx] = p
            Recall_iou_thres[..., tidx] = r
            AP_iou_thres[..., tidx] = ap

        return Precision_iou_thres, Recall_iou_thres, AP_iou_thres, ap_class, self.iou_thres

    def _predict_pil(self, pil_img, **kwargs):
        '''
        Args:
            pil_img: PIL.Image.Image
            input_size: int, input resolution
            conf_thres: float, confidence threshold
        '''
        input_size = kwargs.get('input_size', self.input_size)
        conf_thres = kwargs.get('conf_thres', self.conf_thres)
        nms_thres = kwargs.get('nms_thres', self.nms_thres)
        pro_type = kwargs['pro_type'] if 'pro_type' in kwargs else None
        # assert isinstance(pil_img, Image.Image), 'input must be a PIL.Image'
        assert input_size is not None, 'Please specify the input resolution'
        assert conf_thres is not None, 'Please specify the confidence threshold'

        # pad to square
        input_img, _, pad_info = utils.rect_to_square(pil_img, None, input_size, pro_type, 0)
        if pro_type=='torch':
            input_ori = tvf.to_tensor(input_img)
            input_ = input_ori.unsqueeze(0)
        else:
            input_ori = torch.from_numpy(input_img).permute((2, 0, 1)).float() / 255.
            input_ = input_ori.unsqueeze(0)

        assert input_.dim() == 4
        input_ = input_.cuda()
        with torch.no_grad():
            dts = self.model(input_).cpu()

        dts = dts.squeeze()
        # post-processing
        dts = dts[dts[:, 4:].max(-1)[0] >= conf_thres]
        if len(dts):
            dts = utils.non_max_suppression(dts, conf_thres, nms_thres)
            dts = utils.detection2original(dts, pad_info.squeeze())
        return dts
