import torch
import torch.nn as nn
import torch.nn.functional as F


def roi_pooling(feature_map, rois, size=(7, 7)):
    """
    :param feature_map: (1, C, H, W)
    :param rois: (1, N, 4) N refers to bbox num, 4 represent (ltx, lty, w, h)
    :param size: output size
    :return: (1, C, size[0], size[1])
    """
    output = []
    rois_num = rois.size(1)

    for i in range(rois_num):
        roi = rois[0][i]
        x, y, w, h = roi
        output.append(F.adaptive_max_pool2d(feature_map[:, :, y:y+h, x:x+w], size))

    return torch.cat(output, 1)
def making_rois(feature_map, spliting = [16, 4, 2]):
    f_B, f_C, f_H, f_W = feature_map.shape
    rois = []
    for s in spliting:
        x_s = torch.arange(0, f_W - 1, s)
        y_s = torch.arange(0, f_H-1, s)
        for y in y_s:
            for x in x_s:
                roi = [x, y, s, s]
                rois.append(roi)
    rois = torch.tensor(rois)
    rois = rois.view(1, -1, 4)
    return rois

def Conv(input, out_channel, kernel_size):
    B, C, H, W = input.shape
    conv = nn.Conv2d(C, out_channel, kernel_size)
    output = conv(input)
    return output