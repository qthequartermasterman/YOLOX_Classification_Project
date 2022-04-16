import cv2
from pathlib import Path

import numpy as np
import torch
from torch import nn

from yolox.data_augment import preproc
from yolox.yolox import YOLOX, get_model


# Load model
class dotdict(dict):
    """
    Dotdict is just a dictionary whose elements can be referenced with a dot operation.
    I.e. dotdict['x'] == dotdict.x

    This is useful because the original YOLOX used a custom class to hold a lot of extra configuration that
    we do not need.
    """
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

# Load Images
img_dir = 'imgs/'
images = [cv2.imread(str(im)) for im in Path(img_dir).glob('*.jpg')]
print(f'There are {len(images)} images')
inp_imgs = np.zeros([len(images), 3, opt.test_size[0], opt.test_size[1]], dtype=np.float32)
for b_i, image in enumerate(images):
    img, r = preproc(image, opt.test_size, opt.rgb_means, opt.std)
    inp_imgs[b_i] = img


with torch.no_grad():
    inp_imgs = torch.from_numpy(inp_imgs).to(opt.device)
    yolo_outputs = model(inp_imgs)
    # print(yolo_outputs)
    print(len(yolo_outputs))
    print([t.shape for t in yolo_outputs])
