import cv2
from pathlib import Path

import numpy as np
import torch
from torch import nn

from data_augment import preproc
from models.backbone import CSPDarknet
from models.neck.yolo_fpn import YOLOXPAFPN


class dotdict(dict):
    def __getattr__(self, x):
        return self['x']


opt = dotdict()
opt.input_size = (640, 640)
opt.random_size = (10, 20)  # None; multi-size train: from 448(14*32) to 832(26*32), set None to disable it
opt.test_size = (640, 640)
opt.rgb_means = [0.485, 0.456, 0.406]
opt.std = [0.229, 0.224, 0.225]
opt.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
opt.backbone = "CSPDarknet-nano"
opt.depth_wise = True
opt.use_amp = False  # True, Automatic mixed precision


# Load model

class IdentityModule(nn.Module):
    def forward(self, x):
        return x

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03


def get_model(opt):
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

    head = IdentityModule()
    loss = IdentityModule()

    # define network
    model = YOLOX(opt, backbone=backbone, neck=neck, head=head, loss=loss)
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


model = get_model(opt)

# Load Images
img_dir = 'imgs/'
images = [cv2.imread(str(im)) for im in Path(img_dir).glob('*.jpg')]

with torch.no_grad():
    img_ratios, img_shape = [], []
    inp_imgs = np.zeros([len(images), 3, opt.test_size[0], opt.test_size[1]], dtype=np.float32)
    for b_i, image in enumerate(images):
        img_shape.append(image.shape[:2])
        img, r = preproc(image, opt.test_size, opt.rgb_means, opt.std)
        inp_imgs[b_i] = img
        img_ratios.append(r)

    inp_imgs = torch.from_numpy(inp_imgs).to(opt.device)
    yolo_outputs = model(inp_imgs)
    # print(yolo_outputs)
    print(len(yolo_outputs))
    print([t.shape for t in yolo_outputs])
    # predicts = yolox_post_process(yolo_outputs, self.opt.stride, self.opt.num_classes, vis_thresh,
    #                               self.opt.nms_thresh, self.opt.label_name, img_ratios, img_shape)
