import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import argparse
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
from torch.utils.data import DataLoader

from utils_scrpits.datasets_crop import Dataset4YoloAngle
from utils_scrpits.parse_config import *
from utils_scrpits import timer, logger, Detect
from utils_scrpits.utils import *
from utils_scrpits.get_module_list import *
import datetime
import time
from terminaltables import AsciiTable
from collections import OrderedDict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--input_size', type=int, default=(640, 640))

    parser.add_argument("--conf_thres", type=float, default=0.5, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou thresshold for compute AP")

    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--data_config', type=str, default='cfg/cls.names')
    parser.add_argument('--model_def', type=str, default='cfg/model.cfg')
    parser.add_argument('--eval_interval', type=int, default=1000)
    parser.add_argument('--img_interval', type=int, default=500)
    parser.add_argument('--print_interval', type=int, default=1)
    parser.add_argument('--checkpoint_interval', type=int, default=10)

    parser.add_argument("--n_cpu", type=int, default=8,
                        help="number of cpu threads to use during batch generation")
    parser.add_argument('--preprocess_type', type=str, default='cv2', choices=['cv2', 'torch'],
                        help='image preprocess type')
    parser.add_argument('--enable_aug', type=bool, default=False, help='enable data augmentation or not')
    parser.add_argument('--visulization', type=bool, default=False, help='visualize the training data bofore forward')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # -------------------------- settings ---------------------------
    # assert not args.adversarial
    job_name = '{}'.format(datetime.datetime.now().strftime('%b%d-%H'))
    batch_size = args.batch_size
    num_cpu = 0 if batch_size == 1 else 4
    subdivision = 128 // batch_size
    # SGD optimizer
    decay_SGD = 0.0005 * batch_size * subdivision

    # dataset setting
    print('INFO: Initialing train and test dataset...')

    train_img_dir = ['/home/jetson/Crack_Detect-main/train/JPEGImages']
    train_json = ['/home/jetson/Crack_Detect-main/shale_train_2021.json']

    val_img_dir = ['/home/jetson/Crack_Detect-main/train/JPEGImages']
    val_json = ['/home/jetson/Crack_Detect-main/shale_train_2021.json']
    # read the class name
    class_names = parse_data_config(args.data_config)['names'].split(',')

    # lr_SGD = 0.0001 / batch_size / subdivision
    lr_SGD = 0.0001


    # Learning rate setup
    def burnin_schedule(i):
        burn_in = 500
        if i < burn_in:
            factor = (i / burn_in) ** 2
        elif i < 10000:
            factor = 1.0
        elif i < 20000:
            factor = 0.3
        else:
            factor = 0.1
        return factor

    dataset = Dataset4YoloAngle(train_img_dir, train_json, args.input_size, args.enable_aug, pro_type=args.preprocess_type)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            num_workers=args.n_cpu, pin_memory=True, drop_last=False)
    dataiterator = iter(dataloader)

    module_defs = parse_model_config(args.model_def)
    model = RotateDetectNet(module_defs)
    model=model.to(device)

    # =================对初始化模型进行赋值操作(配置文件发生变化)================
    # 注：如果修改了cls.name以及model.cfg配置文件则需要将此段注释全部去掉
    #model_dict = model.state_dict()

    #start_iter = -1
    #if args.checkpoint:
        #print("loading ckpt...", args.checkpoint)
        #weights_path = os.path.join('./weights/', args.checkpoint)
        #pretrained_dict = torch.load(weights_path)['model']


        #new_state_dict = OrderedDict()
        #for k, v in pretrained_dict.items():
            #if k.split('.')[1] not in ['81', '93', '105']:
                #new_state_dict[k] = v
        #model_dict.update(new_state_dict)
        #model.load_state_dict(model_dict)
        #model.to(device)
    # ===================对初始化模型进行赋值操作(配置文件发生变化)==============


    # # # =================(如果配置文件未发生变化)================
    start_iter = -1
    if args.checkpoint:
        print("INFO: loading ckpt %s ..."%args.checkpoint)
        weights_path = os.path.join('./weights/', args.checkpoint)
        state = torch.load(weights_path)
    #
        model.load_state_dict(state['model'])
        start_iter = state['iter']
        model.to(device)
    # # # =================(如果配置文件未发生变化)================

    logger = logger.Logger('./logs/%s'%job_name)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr_SGD, momentum=0.9, dampening=0,
                                weight_decay=decay_SGD)

    # optimizer.load_state_dict(state['optimizer'])
    print('INFO: Begin from iteration: %d ...'%start_iter)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, burnin_schedule)
    scheduler.last_epoch = start_iter - 1

    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, burnin_schedule, last_epoch=-1)

    # start training loop
    best_value = 0
    start_time = timer.tic()
    for iter_i in range(start_iter, 390):
        # evaluation
        if iter_i % args.eval_interval == 0 and iter_i > 1:
            with timer.contexttimer() as t0:
                model.eval()

                model_eval = Detect.Detector(model=model, conf_thres=args.conf_thres, nms_thres=args.nms_thres,
                                             iou_thres=args.iou_thres, input_size=args.input_size,
                                             class_name=class_names)
                metrics_output = model_eval.forward(val_img_dir, gt_path=val_json, input_size=args.input_size,
                                                    pro_type=args.preprocess_type)

                if metrics_output is not None:
                    precision, recall, AP, ap_class, iou_thres = metrics_output
                    evaluation_metrics = [
                        ("validation/precision", precision.mean()),
                        ("validation/recall", recall.mean()),
                        ("validation/mAP", AP.mean()),
                    ]
                    logger.list_of_scalars_summary(evaluation_metrics, iter_i)

                    # Print class APs and mAP
                    ap_table = [
                        ["Index", "Class name", "Precision(0.5)", "Recall(0.5)", "AP(0.5)", "AP(0.75)", "AP(0.5:0.95)"]]
                    for i, c in enumerate(ap_class):
                        ap_table += [
                            [c, class_names[c], "%.5f" % precision[c][0], "%.5f" % recall[0][i], "%.5f" % AP[c][0],
                             "%.5f" % AP[c][iou_thres.index(0.75)], "%.5f" % AP[c].mean()]]
                    print(AsciiTable(ap_table).table)

                    with open('./logs/%s/log.txt'%job_name, 'a') as fd:
                        fd.write('Iteration: %d\n' % iter_i + AsciiTable(ap_table).table + '\n')

            model.train()

        optimizer.zero_grad()
        for inner_iter_i in range(subdivision):
            start = time.time()
            try:
                imgs, targets, cats, _, _ = next(dataiterator)  # load a batch
            except StopIteration:
                dataiterator = iter(dataloader)
                imgs, targets, cats, _, _ = next(dataiterator)  # load a batch
            if args.visulization:
                for index, img in enumerate(imgs):
                    target = targets[index]
                    np_img = (img.data.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                    np_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
                    boxes = target.numpy()
                    for box in boxes:
                        if sum(box) == 0:
                            continue
                        x = box[0] * np_img.shape[1]
                        y = box[1] * np_img.shape[0]
                        w = box[2] * np_img.shape[1]
                        h = box[3] * np_img.shape[0]
                        x1 = x - w/2
                        y1 = y - h/2
                        x2 = x + w/2
                        y2 = y + h/2
                        cv2.rectangle(np_img, (int(x1), int(y1)), (int(x2), int(y2)), color=(0, 255, 255), thickness=1, lineType=cv2.LINE_4)
                    cv2.namedWindow('1', cv2.WINDOW_NORMAL)
                    cv2.imshow('1', np_img)
                    cv2.waitKey()

            imgs = imgs.cuda()
            torch.cuda.reset_max_memory_allocated()
            loss = model(imgs, targets, labels_cats=cats)
            loss.backward()
        optimizer.step()
        scheduler.step()

        # logging
        if iter_i % args.print_interval == 0:
            sec_used = timer.tic() - start_time
            time_used = timer.sec2str(sec_used)
            avg_iter = timer.sec2str(sec_used / (iter_i + 1 - start_iter))
            avg_epoch = avg_iter / batch_size / subdivision * 118287
            print('\nTotal time: {}, iter: {}, epoch: {}'.format(time_used, avg_iter, avg_epoch))
            # current_lr = scheduler.get_lr()[0] * batch_size * subdivision
            current_lr = scheduler.get_lr()[0]
            print('[Iteration {}] [learning rate {}]'.format(iter_i, '%.3f' % current_lr),
                  '[Total loss {}] [img size {}]'.format('%.2f' % loss, dataset.img_size))
            print(model.loss_str)
            max_cuda = torch.cuda.max_memory_allocated(0) / 1024 / 1024 / 1024
            print('Max GPU memory usage: {} GigaBytes'.format(max_cuda))
            torch.cuda.reset_max_memory_allocated(0)

        # save checkpoint
        if iter_i > 0 and (iter_i % args.checkpoint_interval == 0):
            state_dict = {
                'iter': iter_i,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            save_path = os.path.join('./weights', 'CrackDetection_{}_{}.pth'.format(job_name, iter_i))
            torch.save(state_dict, save_path)

