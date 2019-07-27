from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

import os
import sys
import time
import datetime
import argparse
import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim


def evaluate(model, path, iou_thres, conf_thres, nms_thres, img_size, batch_size):
    model.eval()

    # Get dataloader
    dataset = ListDataset(path, img_size=img_size, augment=False, multiscale=False)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn
    )

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):

        # Extract labels
        labels += targets[:, 1].tolist()
        # Rescale target
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size

        imgs = Variable(imgs.type(Tensor), requires_grad=False)

        with torch.no_grad():
            outputs = model(imgs)
            outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)

        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)

    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    return precision, recall, AP, f1, ap_class

import numpy as np
import cv2
from numpy import float32
from datetime import datetime


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def test_one_image(model, path, iou_thres, conf_thres, nms_thres, img_size, batch_size, use_gpu, yolov3_lite):
    model.eval()  # must be call

    Tensor = torch.cuda.FloatTensor if use_gpu else torch.FloatTensor

    img = cv2.imread("./data/samples/dog.jpg")
    img = cv2.resize(img, (416,416))

    src=img

    # cv2.rectangle(src, (10,10),(100,100), (255,0,0), 1)
    # cv2.imshow("x", src);
    # cv2.waitKey(0);

    img=img.reshape(-1, img.shape[0], img.shape[1], img.shape[2]).astype(float32)
    img = img/255.0

    # for i in range():
    #     img=np.concatenate((img, img), axis=0)
    # img=np.concatenate((img, img), axis=0)

    print(img.shape)
    img = np.transpose(img,axes=(0, 3, 1, 2)) 
    print(img.shape)

    # images
    images = torch.from_numpy(img).cuda() if use_gpu else torch.from_numpy(img).cpu()

    print(type(images))
    print(images.shape)

    outputs = model(images)

    dt0 = time.process_time() # statistic process time
    # for i in range(1000):
    #     outputs = model(images)
    outputs = model(images)

    dt1 = time.process_time() 
    print("process time =", dt1 - dt0, "s")

    if yolov3_lite:
        print(len(outputs[0][0]))  # output 10647 rectangle
        print(outputs[0][0])  # output 10647 rectangle
        # exit(0)
        conf_thres = 0.8
        # nms_thres=0.1
        outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)
        # Output 7 Dim() (x,y,w,h,confidence,...)  
        print(len(outputs))
        print(outputs)
        # for rslt in outputs:
        # for rt in outputs[0]:
        #     cv2.rectangle(src, (rt[0], rt[1]),(rt[2], rt[3]), (255,0,0), 2)
        # cv2.imshow("x", src);
        # cv2.waitKey(0);
    else:
        # print(len(outputs[0]))  # output 10647 rectangle
        conf_thres = 0.8
        outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)
        # Output 7 Dim() (x,y,w,h,confidence,...)  
        # print(len(outputs[0]))
        # for rslt in outputs:
        for rt in outputs[0]:
            cv2.rectangle(src, (rt[0], rt[1]),(rt[2], rt[3]), (255,0,0), 2)
        cv2.imshow("x", src);
        cv2.waitKey(0);


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/coco.data", help="path to data config file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.001, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.5, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    opt = parser.parse_args()
    print(opt)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    device = None
    use_gpu = torch.cuda.is_available()
    # use_gpu = False
    if use_gpu:
        device = "cuda"
        print("Support GPU")
    else:
        device = "cpu"
        print("Support CPU")

    data_config = parse_data_config(opt.data_config)
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])

    # Initiate model
    print("Initiate model ...")
    yolov3_lite = False
    opt.model_def = "./config/yolov3-tiny.cfg" if yolov3_lite else "config/yolov3.cfg"
    model = Darknet(opt.model_def).to(device)
    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))

    # print("Compute mAP...")

    # precision, recall, AP, f1, ap_class = evaluate(
    #     model,
    #     path=valid_path,
    #     iou_thres=opt.iou_thres,
    #     conf_thres=opt.conf_thres,
    #     nms_thres=opt.nms_thres,
    #     img_size=opt.img_size,
    #     batch_size=8,
    # )

    test_one_image(model,
        path=valid_path,
        iou_thres=opt.iou_thres,
        conf_thres=opt.conf_thres,
        nms_thres=opt.nms_thres,
        img_size=opt.img_size,
        batch_size=1,
        use_gpu=use_gpu,
        yolov3_lite=yolov3_lite,
    )

    # print("Average Precisions:")
    # for i, c in enumerate(ap_class):
    #     print("f+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")

    # print("mAP: {AP.mean()}")
