import torch
import torch.nn as nn
import src.net.backbones.resnet as resnet
import src.net.db_detector as db_detector
import math
import cv2


class model(nn.Module):
    seg_detrctor = db_detector

    def __init__(self, fpn):
        super(model, self).__init__()
        self.fpn = fpn
        self.load_fpn()
        self.fpn_net = None

    def load_fpn(self):
        if self.fpn == 'resnet50':
            self.fpn_net = resnet.resnet50()

    def resize_img(self,img):
        height, width, _ = img.shape
        if height < width:
            new_height = 736
            new_width = int(math.ceil(new_height / height * width / 32) * 32)
        else:
            new_width = 736
            new_height = int(math.ceil(new_width / width * height / 32) * 32)
        resized_img = cv2.resize(img, (new_width, new_height))
        return resized_img