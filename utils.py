import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def intesection_over_union(self, box_1, box_2, box_format):
    """"
    Arguments:
        box_1 : (X1,Y1,X2,Y2) of shape (N,4)
        box_2 : (X1,Y1,X2,Y2) of shape (N,4)
    Return :
     IOU
    """
    if box_format == 'midpoint':
        x1 = torch.max((box_1[:, 0] - box_1[:, 2]) / 2, (box_2[:, 0] - box_2[:, 2]) / 2)
        y1 = torch.max((box_1[:, 1] - box_1[:, 3]) / 2, (box_2[:, 1] - box_2[:, 3]) / 2)
        x2 = torch.max((box_1[:, 0] + box_1[:, 2]) / 2, (box_2[:, 0] + box_2[:, 2]) / 2)
        y2 = torch.max((box_1[:, 1] + box_1[:, 3]) / 2, (box_2[:, 1] + box_2[:, 3]) / 2)

    if box_format == 'corners':
        x1 = torch.max(box_1[:,0], box_2[:,0])
        y1 = torch.max(box_1[:,1], box_2[:,1])
        x2 = torch.min(box_1[:,2], box_2[:,2])
        y2 = torch.min(box_1[:,3], box_2[:,3])

    intersection = (x2-x1).clamp(0) * (y2-y1).clamp(0)
    # .clamp(0) for condition when they do not intersect


    box_1_area = abs((box_1[:,3] - box_1[:,1]) * (box_1[:,2] - box_1[:,0]))
    box_2_area = abs((box_2[:,3] - box_2[:,1]) * (box_2[:,2] - box_2[:,0]))
    union = box_1_area + box_2_area - intersection + 1e-6

    IoU = intersection / union


def nms(bboxes, iou_threshold, threshold ):
    """
    Arguments:
    param bboxes:
    param iou_threshold:
    param threshold:
    Return:

    """
    
