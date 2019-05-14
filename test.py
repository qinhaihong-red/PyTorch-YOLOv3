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

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=16, help="size of each image batch")
parser.add_argument("--model_config_path", type=str, default="config/yolov3.cfg", help="path to model config file")
parser.add_argument("--data_config_path", type=str, default="config/coco.data", help="path to data config file")
parser.add_argument("--weights_path", type=str, default="/home/hh/deeplearning_daily/darknet_src/darknet/yolov3.weights", help="path to weights file")
parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
parser.add_argument("--conf_thres", type=float, default=0.5, help="object confidence threshold")
parser.add_argument("--nms_thres", type=float, default=0.45, help="iou thresshold for non-maximum suppression")
parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
parser.add_argument("--use_cuda", type=bool, default=True, help="whether to use cuda if available")
opt = parser.parse_args()
print(opt)

cuda = torch.cuda.is_available() and opt.use_cuda

# Get data configuration
data_config = parse_data_config(opt.data_config_path)
test_path = data_config["valid"]#这里的验证集是从原著中5k数据集中随机选择的1k,为了加速检测时间
num_classes = int(data_config["classes"])#80

# Initiate model
model = Darknet(opt.model_config_path)
model.load_darknet_weights(opt.weights_path)

if cuda:
    model = model.cuda()

model.eval()

# Get dataloader
#path to this:/home/hh/dataset/coco/5k.txt 或者 1k.txt
dataset = ListDataset(test_path) #返回填充、变换过的图像和标签张量
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)#dataloader和dataset分离

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

print("Compute mAP...")

all_detections = []
all_annotations = []

#迭代批数据，输出预测以及对应的真实标签
for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):#每批数据：(16,3,416,416),(16,50,5)

    imgs = Variable(imgs.type(Tensor))#16,3,416,416

    with torch.no_grad():
        outputs = model(imgs)#原始输出：16,10647,85
        outputs = non_max_suppression(outputs, 80, conf_thres=opt.conf_thres, nms_thres=opt.nms_thres)#conf_thres和nms处理后的输出列表：[16,xx,7]
    #对批里面的每个图像的预测输出和对应标签，都逐一处理
    for output, annotations in zip(outputs, targets):#每次迭代的数据：(xx,7),(50,5)，每个都是二维的
        #为每个图像的预测输出都建立1个80分类统计的列表
        all_detections.append([np.array([]) for _ in range(num_classes)])
        if output is not None:
            # Get predicted boxes, confidence scores and labels
            pred_boxes = output[:, :5].cpu().numpy()#n,5
            scores = output[:, 4].cpu().numpy()#n,
            pred_labels = output[:, -1].cpu().numpy()#n,

            # Order by confidence
            sort_i = np.argsort(scores)
            pred_labels = pred_labels[sort_i]
            pred_boxes = pred_boxes[sort_i]

            for label in range(num_classes):
                all_detections[-1][label] = pred_boxes[pred_labels == label] #把预测类别为pred_labels的box，分给label. np:(n,5)

        all_annotations.append([np.array([]) for _ in range(num_classes)])#为每个图像建立对应的真实标签
        # annotations (50,5)
        if any(annotations[:, -1] > 0):

            annotation_labels = annotations[annotations[:, -1] > 0, 0].numpy()#np:(n,)
            _annotation_boxes = annotations[annotations[:, -1] > 0, 1:]#tensor(n,4)

            # Reformat to x1, y1, x2, y2 and rescale to image dimensions
            # 把标签的框坐标，由中心-宽高 变换为 左上右下 形式，再缩放到416尺度
            annotation_boxes = np.empty_like(_annotation_boxes)#np. 下面tensor可以直接转ndarray
            annotation_boxes[:, 0] = _annotation_boxes[:, 0] - _annotation_boxes[:, 2] / 2
            annotation_boxes[:, 1] = _annotation_boxes[:, 1] - _annotation_boxes[:, 3] / 2
            annotation_boxes[:, 2] = _annotation_boxes[:, 0] + _annotation_boxes[:, 2] / 2
            annotation_boxes[:, 3] = _annotation_boxes[:, 1] + _annotation_boxes[:, 3] / 2
            annotation_boxes *= opt.img_size #放大为相对于416*416

            for label in range(num_classes):
                all_annotations[-1][label] = annotation_boxes[annotation_labels == label, :]#np:(n,4)

average_precisions = {}
#对80个类别进行迭代，分别计算每个类别的AP
for label in range(num_classes):
    true_positives = []
    scores = []
    num_annotations = 0 #记录每个类别真实标签的数量
    #对当前类别，迭代所有图片
    for i in tqdm.tqdm(range(len(all_annotations)), desc=f"Computing AP for class '{label}'"):
        detections = all_detections[i][label] #  n,5
        annotations = all_annotations[i][label]# m,4

        num_annotations += annotations.shape[0]
        detected_annotations = []
        #对此图片detection中关于当前label类别的所有n个预测，进行逐一处理
        for *bbox, score in detections:
            scores.append(score)
            
            #当前图像没有关于当前类别的标签，但是detections中却预测到了，这是一个false_positive，用0来表示
            if annotations.shape[0] == 0:
                true_positives.append(0)
                continue
            #annotation是对应本图当前类别label的m个标签，由于bbox是(1,4),因此返回的overlaps是:np(1,m)
            overlaps = bbox_iou_numpy(np.expand_dims(bbox, axis=0), annotations)
            assigned_annotation = np.argmax(overlaps, axis=1)
            max_overlap = overlaps[0, assigned_annotation]
            #选出最大的overlap与iou_thres对比，作为true_positive，并且记录. 注意重复的框算作false_positive
            if max_overlap >= opt.iou_thres and assigned_annotation not in detected_annotations:
                true_positives.append(1)
                detected_annotations.append(assigned_annotation)
            else:
                true_positives.append(0)

    # no annotations -> AP for this class is 0
    # 在所有图像中都没有发现这个类别的真实标签，则此类别的AP记为0.
    if num_annotations == 0:
        average_precisions[label] = 0
        continue

    #对当前类别计算AUC AP
    #>1.根据置信对tp和fp排序
    #>2.计算tp和fp的累加和
    #>3.计算recall和precision
    #>4.根据recall和precision计算AUC AP
    true_positives = np.array(true_positives)
    false_positives = np.ones_like(true_positives) - true_positives
    # sort by score
    indices = np.argsort(-np.array(scores))#-表示降序
    false_positives = false_positives[indices]
    true_positives = true_positives[indices]

    # compute false positives and true positives
    false_positives = np.cumsum(false_positives)
    true_positives = np.cumsum(true_positives)

    # compute recall and precision
    recall = true_positives / num_annotations
    precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

    # compute average precision
    average_precision = compute_ap(recall, precision)
    average_precisions[label] = average_precision

print("Average Precisions:")
for c, ap in average_precisions.items():
    print(f"+ Class '{c}' - AP: {ap}")

#计算mAP
mAP = np.mean(list(average_precisions.values()))
print(f"mAP: {mAP}")
