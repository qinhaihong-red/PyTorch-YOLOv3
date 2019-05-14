from __future__ import division
import math
import time
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def to_cpu(tensor):
    return tensor.detach().cpu()


def load_classes(path):
    """
    Loads class labels at 'path'
    """
    fp = open(path, "r")
    names = fp.read().split("\n")[:-1]
    return names


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

##关于AUC AP的计算说明，参考：
##http://note.youdao.com/noteshare?id=2c24d33f527df44b2c057cffd7954ba1
def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    m_recall    =  np.concatenate(([0.0], recall, [1.0]))
    m_precision =  np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(m_precision.size - 1, 0, -1):
        m_precision[i - 1] = np.maximum(m_precision[i - 1], m_precision[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(m_recall[1:] != m_recall[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((m_recall[i + 1] - m_recall[i]) * m_precision[i + 1])
    return ap


def get_batch_statistics(outputs, targets, iou_threshold):
    """ Compute true positives, predicted scores and predicted labels per sample """
    batch_metrics = []
    for sample_i in range(len(outputs)):

        if outputs[sample_i] is None:
            continue

        output = outputs[sample_i]
        pred_boxes = output[:, :4]
        pred_scores = output[:, 4]
        pred_labels = output[:, -1]

        true_positives = np.zeros(pred_boxes.shape[0])

        annotations = targets[targets[:, 0] == sample_i][:, 1:]
        target_labels = annotations[:, 0] if len(annotations) else []
        if len(annotations):
            detected_boxes = []
            target_boxes = annotations[:, 1:]

            for pred_i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):

                # If targets are found break
                if len(detected_boxes) == len(annotations):
                    break

                # Ignore if label is not one of the target labels
                if pred_label not in target_labels:
                    continue

                iou, box_index = bbox_iou(pred_box.unsqueeze(0), target_boxes).max(0)
                if iou >= iou_threshold and box_index not in detected_boxes:
                    true_positives[pred_i] = 1
                    detected_boxes += [box_index]
        batch_metrics.append([true_positives, pred_scores, pred_labels])
    return batch_metrics


def bbox_wh_iou(wh1, wh2):
    wh2 = wh2.t()
    w1, h1 = wh1[0], wh1[1]
    w2, h2 = wh2[0], wh2[1]
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
    return inter_area / union_area


def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.4):
    """
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """

    # From (center x, center y, width, height) to (x1, y1, x2, y2)
    # nms干的第一件事就是把框坐标由 中心-宽高 变为 左上右下
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]# 返回列表，长度等同批的长度
    #prediction(16,10647,85)包括一批图像，下面逐一迭代处理
    for image_i, image_pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        conf_mask = (image_pred[:, 4] >= conf_thres).squeeze()
        image_pred = image_pred[conf_mask] #(n,85)
        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(image_pred[:, 5 : 5 + num_classes], 1, keepdim=True)#选出每一行的最大得分分类以及预测标签，得到一列数据
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred) # class_conf, class_pred 的形状都是(n,1)
        detections = torch.cat((image_pred[:, :5], class_conf.float(), class_pred.float()), 1)#在列方向拼接：(n,7)
        # Iterate through all predicted classes
        unique_labels = detections[:, -1].cpu().unique()
        if prediction.is_cuda:
            unique_labels = unique_labels.cuda()

        for c in unique_labels:
            # Get the detections with the particular class
            detections_class = detections[detections[:, -1] == c] #选出c类的所有预测：(_c,7)
            # Sort the detections by maximum objectness confidence
            _, conf_sort_index = torch.sort(detections_class[:, 4], descending=True)#对这些属于c类的所有预测按置信降序排列
            detections_class = detections_class[conf_sort_index] #(_c,7)
            # Perform non-maximum suppression
            max_detections = []
            while detections_class.size(0):
                # Get detection with highest confidence and save as max detection
                max_detections.append(detections_class[0].unsqueeze(0))#先把置信最大的预测加入列表
                # Stop if we're at the last detection
                if len(detections_class) == 1:
                    break
                # Get the IOUs for all boxes with lower confidence
                #把其他的预测与最大置信的预测做IOU比较，进行nms处理
                ious = bbox_iou(max_detections[-1], detections_class[1:]) #传入:(1,7),(_c,7) 传出:(1,_c)
                # Remove detections with IoU >= NMS threshold
                detections_class = detections_class[1:][ious < nms_thres]

            max_detections = torch.cat(max_detections).data #(_n,7):max_detections是个列表，把这个列表合并在一起，作为c类nms处理后的结果
            # Add max detections to outputs
            output[image_i] = (
                max_detections if output[image_i] is None else torch.cat((output[image_i], max_detections))
            )#批中的每个图像，都有自己的经过nms处理后的预测tensor

    return output


def build_targets(
    pred_boxes, pred_conf, pred_cls, target, anchors, num_anchors, num_classes, grid_size, ignore_thres, img_dim
):

    #pred_boxes:(batch,3,grid,grid,4)，经过 中心-位移加法 和 宽高-指数、乘法 缩放，尺度相对于grid
    #target:    (batch,50,5),框坐标形式为 中心-宽高 比例，相对尺度为原图经过pad的尺寸，而不是416
    nB = target.size(0)
    nA = num_anchors #3
    nC = num_classes #80
    nG = grid_size   #grid
    mask = torch.zeros(nB, nA, nG, nG) # (batch,3,grid,grid)
    conf_mask = torch.ones(nB, nA, nG, nG)
    tx = torch.zeros(nB, nA, nG, nG)
    ty = torch.zeros(nB, nA, nG, nG)
    tw = torch.zeros(nB, nA, nG, nG)
    th = torch.zeros(nB, nA, nG, nG)
    tconf = torch.ByteTensor(nB, nA, nG, nG).fill_(0)
    tcls = torch.ByteTensor(nB, nA, nG, nG, nC).fill_(0)

    nGT = 0
    nCorrect = 0
    #迭代批中的每个图像
    for b in range(nB):
        #迭代图像中可能存在的每个框
        for t in range(target.shape[1]):
            if target[b, t].sum() == 0:
                continue
            nGT += 1
            # Convert to position relative to box
            # 把gt的 中心-宽高 比例坐标，缩放为相对于grid尺度
            gx = target[b, t, 1] * nG
            gy = target[b, t, 2] * nG
            gw = target[b, t, 3] * nG
            gh = target[b, t, 4] * nG
            
            # Get grid box indices
            gi = int(gx) #表示列，用它表示下标在后
            gj = int(gy) #表示行，用它表示下标在前
            
            # Get shape of gt box
            gt_box = torch.FloatTensor(np.array([0, 0, gw, gh])).unsqueeze(0) #(1,4)
            
            # Get shape of anchor box .
            # anchors是经过除stride缩放的.
            # -> (3,4)
            anchor_shapes = torch.FloatTensor(np.concatenate((np.zeros((len(anchors), 2)), np.array(anchors)), 1))
            
            '''
            https://www.analyticsvidhya.com/blog/2018/12/practical-guide-object-detection-yolo-framewor-python/
            :The objects are assigned to the anchor boxes based on the similarity of the bounding boxes and 
            the anchor box shape.

            
            这里的形状IOU,主要用于训练阶段挑选合适的anchor. 不同于检测阶段用于nms的IOU.

            '''
            # Calculate iou between gt and anchor shapes
            # gt坐标是(1,0,0,gw,gh).anchor_shape是(3,0,0,anchor_w,anchor_h).两者都是 左上右下 的坐标形式
            anch_ious = bbox_iou(gt_box, anchor_shapes)#输出为(1,3)

            # Where the overlap is larger than threshold set mask to zero (ignore)
            # (batch,3,grid,grid)，初始值全部为1
            conf_mask[b, anch_ious > ignore_thres, gj, gi] = 0

            # Find the best matching anchor box
            best_n = np.argmax(anch_ious)
            
            # Get ground truth box
            gt_box = torch.FloatTensor(np.array([gx, gy, gw, gh])).unsqueeze(0) #(1,4)
            
            # Get the best prediction
            # pred_boxes是(batch,3,grid,grid,4)，经过 中心-位移加法 和 宽高-指数、乘法 缩放，尺度相对于grid
            pred_box = pred_boxes[b, best_n, gj, gi].unsqueeze(0) #(1,4)
            
            # Masks
            # mask是(batch,3,grid,grid)，初始值全部为0
            mask[b, best_n, gj, gi] = 1
            conf_mask[b, best_n, gj, gi] = 1


            '''
                 # Add offset and scale with anchors
                pred_boxes = FloatTensor(prediction[..., :4].shape)# (batch,3,grid,grid,4)
                pred_boxes[..., 0] = x.data + grid_x #广播
                pred_boxes[..., 1] = y.data + grid_y
                pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
                pred_boxes[..., 3] = torch.exp(h.data) * anchor_h


            '''

            # Coordinates
            # 标签坐标的变换
            tx[b, best_n, gj, gi] = gx - gi
            ty[b, best_n, gj, gi] = gy - gj
            
            # Width and height
            # 标签宽高的变换
            tw[b, best_n, gj, gi] = math.log(gw / anchors[best_n][0] + 1e-16)
            th[b, best_n, gj, gi] = math.log(gh / anchors[best_n][1] + 1e-16)
            
            # One-hot encoding of label
            target_label = int(target[b, t, 0])
            tcls[b, best_n, gj, gi, target_label] = 1 #(batch,3,grid,grid,80),初始值全为0
            tconf[b, best_n, gj, gi] = 1 #(batch,3,grid,grid)，初始值全为0

            # Calculate iou between ground truth and best matching prediction
            iou = bbox_iou(gt_box, pred_box, x1y1x2y2=False) #(1,)

            #pred_cls:(batch,3,grid,grid,80)
            pred_label = torch.argmax(pred_cls[b, best_n, gj, gi])# (1,)
            
            #pred_conf:(batch,3,grid,grid)
            score = pred_conf[b, best_n, gj, gi] #(1,)

            if iou > 0.5 and pred_label == target_label and score > 0.5:
                nCorrect += 1

    return nGT, nCorrect, mask, conf_mask, tx, ty, tw, th, tconf, tcls


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return torch.from_numpy(np.eye(num_classes, dtype="uint8")[y])
