# Time: 2023/12/19 20:38
# Author: Yiming Ma
# Place: Shenzhen
import argparse
import os
from PIL import Image

import torch
from torchvision import transforms

import models
from utils import make_coord
from test import batched_predict

# python demo.py --resolution 256,256 --gpu 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='C:/Users/28958/Desktop/网络部署/liif-single-software/test-image/4-1-LR/1.jpeg')
    parser.add_argument('--model', default='C:/Users/28958/Desktop/网络部署/liif-single-software/checkpoint/epoch-300.pth')
    parser.add_argument('--resolution', default='256,256')
    parser.add_argument('--output', default='output4-1-300.jpeg')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    img = transforms.ToTensor()(Image.open(args.input).convert('RGB'))

    model = models.make(torch.load(args.model)['model'], load_sd=True).cuda()

    h, w = list(map(int, args.resolution.split(',')))
    coord = make_coord((h, w)).cuda()
    cell = torch.ones_like(coord)
    cell[:, 0] *= 2 / h
    cell[:, 1] *= 2 / w
    pred = batched_predict(model, ((img - 0.5) / 0.5).cuda().unsqueeze(0),
        coord.unsqueeze(0), cell.unsqueeze(0), bsize=30000)[0]
    pred = (pred * 0.5 + 0.5).clamp(0, 1).view(h, w, 3).permute(2, 0, 1).cpu()
    transforms.ToPILImage()(pred).save(args.output)
