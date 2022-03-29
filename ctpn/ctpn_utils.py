import numpy as np
from config import *


def gen_anchor(featuresize, scale):
    # 获取先验框
    heights = [11, 16, 23, 33, 48, 68, 97, 139, 198, 283]
    widths = [16, 16, 16, 16, 16, 16, 16, 16, 16, 16]

    heights = np.array(heights).reshape(len(heights), 1)
    widths = np.array(widths).reshape(len(widths), 1)
    
    base_anchor = np.array([0, 0, 15, 15])
    # get center
    xt = (base_anchor[0] + base_anchor[2]) * 0.5
    yt = (base_anchor[1] + base_anchor[3]) * 0.5

    x1 = xt - widths * 0.5
    y1 = yt - heights * 0.5
    x2 = xt + widths * 0.5
    y2 = yt + heights * 0.5
    base_anchor = np.hstack((x1, y1, x2, y2))

    h, w = featuresize
    shift_x = np.arange(0, w) * scale
    shift_y = np.arange(0, h) * scale

    anchor = []

    for i in shift_y:
        for j in shift_x:
            anchor.append(base_anchor + [j, i, j, i])

    return np.array(anchor).reshape((-1, 4))
    

def cal_iou(box1, box1_area , boxes2, boxes2_area):
    x1 = np.maximum(box1[0], boxes2[:, 0])
    x2 = np.minimum(box1[2], boxes2[:, 2])
    y1 = np.maximum(box1[1], boxes2[:, 1])
    y2 = np.minimum(box1[3], boxes2[:, 3])

    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    iou = intersection / (box1_area + boxes2_area[:] - intersection[:])
    return iou

def cal_overlaps(boxes1, boxes2):
    area1 = (boxes1[:, 0] - boxes1[:, 2]) * (boxes1[:, 1] - boxes1[:, 3])
    area2 = (boxes2[:, 0] - boxes2[:, 2]) * (boxes2[:, 1] - boxes2[:, 3])

    overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))
    
    for i in range(boxes1.shape[0]):
        overlaps[i][:] = cal_iou(boxes1[i], area1[i], boxes2, area2)

    return overlaps

# 偏移量， 伸缩量
def bbox_transform(anchors, gtboxes):
    regr = np.zeros((anchors.shape[0], 2))
    Cy = (gtboxes[:, 1] + gtboxes[:, 3]) * 0.5
    Cya = (anchors[:, 1] + anchors[:, 3]) * 0.5
    h = gtboxes[:, 3] - gtboxes[:, 1] + 1.0
    ha = anchors[:, 3] - anchors[:, 1] + 1.0

    Vc = (Cy - Cya) / ha
    Vh = np.log(h / ha)

    return np.vstack((Vc, Vh)).transpose()

def cal_rpn(imgsize, featuresize, scale, gtboxes):
    imgh, imgw = imgsize

    # get base anchors
    base_anchor = gen_anchor(featuresize, scale)

    # calculate iou
    overlaps = cal_overlaps(base_anchor, gtboxes)

    # init labels -1 don't care  0 is negative  1 is positive
    labels = np.empty(base_anchor.shape[0])
    labels.fill(-1)
    # 非极大抑制
    gt_argmax_overlaps = overlaps.argmax(axis=0)

    anchor_argmax_overlaps = overlaps.argmax(axis=1)
    anchor_max_overlaps = overlaps[range(overlaps.shape[0]), anchor_argmax_overlaps]

    # 最大重叠大于阈值的设为true
    labels[anchor_max_overlaps > IOU_POSITIVE] = 1
    labels[anchor_max_overlaps < IOU_NEGATIVE] = 0
    # 为实际框重叠度最大的设为true
    labels[gt_argmax_overlaps] = 1

    # only keep anchors inside the image
    outside_anchor = np.where(
        (base_anchor[:, 0] < 0) |
        (base_anchor[:, 1] < 0) |
        (base_anchor[:, 2] >= imgw) |
        (base_anchor[:, 3] >= imgh)
    )[0]

    labels[outside_anchor] = -1

    fg_index = np.where(labels == 1)[0]

    if (len(fg_index) > RPN_POSITIVE_NUM):
        labels[np.random.choice(fg_index, len(fg_index) - RPN_POSITIVE_NUM, replace=False)] = -1

    if not OHEM:
        bg_index = np.where(labels == 0)[0]
        num_bg = RPN_TOTAL_NUM - np.sum(labels == 1)
        if (len(bg_index) > num_bg):
            # print('bgindex:',len(bg_index),'num_bg',num_bg)
            labels[np.random.choice(bg_index, len(bg_index) - num_bg, replace=False)] = -1

    # 找重叠度最大的gt计算bbox
    bbox_targets = bbox_transform(base_anchor, gtboxes[anchor_argmax_overlaps, :])

    return [labels, bbox_targets], base_anchor