import cv2
from pathlib import Path

import numpy as np
import torch
from torch import nn

from data_augment import preproc
from models.backbone import CSPDarknet
from models.neck.yolo_fpn import YOLOXPAFPN
from utils.model_utils import load_model
from yolox import YOLOX, get_model


# Load model
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

model = get_model(opt)
model = load_model(model, 'yolox-nano.pth')

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
