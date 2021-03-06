"""Adapted from https://github.com/zhangming8/yolox-pytorch"""

import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from models.backbone import CSPDarknet
from models.neck.yolo_fpn import YOLOXPAFPN
# from models.head.yolo_head import YOLOXHead
# from models.losses import YOLOXLoss
# from models.post_process import yolox_post_process
from models.ops import fuse_model
from .data_augment import preproc
from .utils.model_utils import load_model


class IdentityModule(nn.Module):
    def forward(self, x):
        return x

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03


def get_model(opt,
              head: nn.Module = None,
              freeze_layers: bool = True,
              use_pretrained: Optional[str] = 'pretrained_models/yolox-nano.pth'):
    """
    Create a custom YOLOX-inspired network with the given options (configuration), head network, and determining whether
    early layers should be frozen/pretrained
    :param opt: dotdict with the necessary configurations
    :param head: nn.Module (network) that will take the YOLOX neck output and turn that into the appropriate task
    :param freeze_layers: True if early layers should be frozen for transfer learning
    :param use_pretrained: str pointing to a pretrained model weight or None
    :return:
    """
    # define backbone
    backbone_cfg = {"nano": [0.33, 0.25],
                    "tiny": [0.33, 0.375],
                    "s": [0.33, 0.5],
                    "m": [0.67, 0.75],
                    "l": [1.0, 1.0],
                    "x": [1.33, 1.25]}
    depth, width = backbone_cfg[opt.backbone.split("-")[1]]  # "CSPDarknet-s"
    in_channel = [256, 512, 1024]
    backbone = CSPDarknet(dep_mul=depth, wid_mul=width, out_indices=(3, 4, 5), depthwise=opt.depth_wise)
    # define neck
    neck = YOLOXPAFPN(depth=depth, width=width, in_channels=in_channel, depthwise=opt.depth_wise)
    # # define head
    # head = YOLOXHead(num_classes=opt.num_classes, reid_dim=opt.reid_dim, width=width, in_channels=in_channel,
    #                  depthwise=opt.depth_wise)
    # # define loss
    # loss = YOLOXLoss(opt.label_name, reid_dim=opt.reid_dim, id_nums=opt.tracking_id_nums, strides=opt.stride,
    #                  in_channels=in_channel)

    head = head or IdentityModule()
    loss = IdentityModule()

    # define network
    model = YOLOX(opt, backbone=backbone, neck=neck, head=head, loss=loss)

    # Load network
    if use_pretrained:
        model = load_model(model, use_pretrained)

    # Freeze Layers
    if freeze_layers:
        for p in model.backbone.parameters():
            p.requires_grad = False
        for p in model.neck.parameters():
            p.requires_grad = False
    return model


class YOLOX(nn.Module):
    def __init__(self, opt, backbone, neck, head, loss):
        super(YOLOX, self).__init__()
        self.opt = opt
        self.backbone = backbone
        self.neck = neck
        self.head = head
        self.loss = loss

        self.backbone.init_weights()
        self.neck.init_weights()
        self.head.init_weights()

    def forward(self, inputs, targets=None, show_time=False):
        with torch.cuda.amp.autocast(enabled=self.opt.use_amp):
            body_feats = self.backbone(inputs)
            neck_feats = self.neck(body_feats)
            yolo_outputs = self.head(neck_feats)
            # print('yolo_outputs:', [[i.shape, i.dtype, i.device] for i in yolo_outputs])  # float16 when use_amp=True

            if targets is not None:
                loss = self.loss(yolo_outputs, targets)
                # for k, v in loss.items():
                #     print(k, v, v.dtype, v.device)  # always float32

        return (yolo_outputs, loss) if targets is not None else yolo_outputs


class Detector(object):
    def __init__(self, cfg):
        self.opt = cfg
        self.opt.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.opt.pretrained = None
        self.model = get_model(self.opt)
        print("Loading model {}".format(self.opt.load_model))
        self.model = load_model(self.model, self.opt.load_model)
        self.model.to(self.opt.device)
        self.model.eval()
        if "fuse" in self.opt and self.opt.fuse:
            print("==>> fuse model's conv and bn...")
            self.model = fuse_model(self.model)

    def run(self, images, vis_thresh, show_time=False):
        batch_img = True
        if np.ndim(images) == 3:
            images = [images]
            batch_img = False

        with torch.no_grad():
            if show_time:
                s1 = time.time()

            img_ratios, img_shape = [], []
            inp_imgs = np.zeros([len(images), 3, self.opt.test_size[0], self.opt.test_size[1]], dtype=np.float32)
            for b_i, image in enumerate(images):
                img_shape.append(image.shape[:2])
                img, r = preproc(image, self.opt.test_size, self.opt.rgb_means, self.opt.std)
                inp_imgs[b_i] = img
                img_ratios.append(r)

            if show_time:
                s2 = time.time()
                print("[pre_process] time {}".format(s2 - s1))

            inp_imgs = torch.from_numpy(inp_imgs).to(self.opt.device)
            yolo_outputs = self.model(inp_imgs, show_time=show_time)

            if show_time:
                s3 = sync_time(inp_imgs)
            predicts = yolox_post_process(yolo_outputs, self.opt.stride, self.opt.num_classes, vis_thresh,
                                          self.opt.nms_thresh, self.opt.label_name, img_ratios, img_shape)
            if show_time:
                s4 = sync_time(inp_imgs)
                print("[post_process] time {}".format(s4 - s3))
        if batch_img:
            return predicts
        else:
            return predicts[0]
