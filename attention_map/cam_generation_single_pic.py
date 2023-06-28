# 按比例划分数据集
# 思路，给定数据集，分别取出
import os
import random
import shutil
import csv
import numpy as np
import argparse  # 1.29添加，使用超参数方法
import imghdr  # 1.29判断是否为图片类型

# All imports
import matplotlib.pyplot as plt

from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, transforms
from torchvision.transforms.functional import normalize, resize, to_pil_image

from PIL import Image
import os
import numpy as np
import sys
from tqdm import tqdm

import pickle
import torch
import torch.utils.data as data

from itertools import permutations

# from randstainna import RandStainNA, HSVJitter # 7.14添加

from torchcam.methods import SmoothGradCAMpp, LayerCAM  # 7.14添加
from torchcam.utils import overlay_mask

# from models import model_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='cam_generation')

# Dataset parameters
parser.add_argument('--dataset', metavar='DIR', default='/home/wagnchogn/Desktop/randstiannav2/crc_dataset/train',
                    help='path to dataset')  # ImageFolder形式
parser.add_argument('--result', metavar='DIR', default='/home/wagnchogn/Desktop/randstiannav2/crc_dataset/cam',
                    help='path to result dataset')
parser.add_argument('--model-cam', metavar='DIR',
                    default='/home/wagnchogn/Desktop/randstiannav2/randstiannav2-master/scripts_CRC/output_back/train/baseline_M_BC_HSVstrong0.5_resnet18/best.pth',
                    help='path to model_cam')  # ImageFolder形式
parser.add_argument('--thresh', type=float, default=0.5,
                    help='model_cam thresh')  # ImageFolder形式

img_batch = 0
save_path_list = []
j = 0


def ProcessFile(origin_path, save_path, model_cam, cam_extractor, thresh):  # 三个参数，第一个为每个类别的所有图像在计算机中的位置
    global img_batch, save_path_list, j

    if os.path.isfile(save_path) or os.path.isdir(origin_path):
        return None

    img = Image.open(origin_path)
    input_tensor = normalize(transforms.ToTensor()(img), [0.5071, 0.4865, 0.4409], [0.2673, 0.2564, 0.2762]).unsqueeze(
        0)

    if j == 0:
        img_batch = input_tensor
        save_path_list.append(save_path)
        j += 1
    elif img_batch.shape[0] < 256:  # 可能存在问题：部分数据没有跑到，最后一轮被忽视了，还是得一张一张跑检查一遍
        img_batch = torch.cat([img_batch, input_tensor], dim=0)
        save_path_list.append(save_path)
        return None
    else:
        out = model_cam(img_batch.cuda())

        cams = cam_extractor(out.argmax(dim=-1).detach().cpu().numpy().tolist(), out)
        cams_list = []
        for i in range(cams[0].shape[0]):
            cams_list += [cams[0][i, :, :]]
        segmaps = [to_pil_image(
            (resize(cam.unsqueeze(0), input_tensor.shape[-2:]).squeeze(0) >= thresh).to(dtype=torch.float32)) for cam in
                   cams_list]
        for i in range(len(segmaps)):
            segmaps[i].save(save_path_list[i])
        img_batch = 0
        j = 0
        save_path_list = []


if __name__ == '__main__':
    args = parser.parse_args()

    # 只需给定file_path、test_rate即可完成整个任务
    # 原始路径+分割比例
    ################################
    file_path = args.dataset  # 初始数据集位置
    result_path = args.result  # 期望result位置是一个文件夹，根目录
    ################################
    checkpoint = torch.load(args.model_cam, map_location=torch.device('cpu'))
    model = checkpoint['model']
    epoch_best = checkpoint['epoch']
    acc_best = checkpoint['acc']
    print('epoch:{} acc:{}'.format(epoch_best, acc_best))
    model.cuda()
    model.eval()
    cam_extractor = SmoothGradCAMpp(model, target_layer='layer3')

    if not os.path.exists(result_path):
        os.makedirs(result_path)
    file_dirs = os.listdir(file_path)
    origion_paths = []
    save_dirs = []
    last_len = 0
    thresh = 0.5
    for class_ in file_dirs:  # 嵌套把所有文件夹建好，同时所有路径弄好
        origin_class_path = os.path.join(file_path, class_)
        result_class_path = os.path.join(result_path, class_)
        ori_imgs = os.listdir(origin_class_path)
        cam_imgs = os.listdir(result_class_path)
        if not os.path.exists(result_class_path):
            os.makedirs(result_class_path)
        for img in tqdm(ori_imgs):
            if img in cam_imgs:
                continue
            #print(img)
            save_path_list = []
            origin_img_path = os.path.join(origin_class_path, img)
            result_img_path = os.path.join(result_class_path, img)
            # origion_paths.append(origin_img_path)
            # save_dirs.append(result_img_path)
            img = Image.open(origin_img_path)
            input_tensor = normalize(transforms.ToTensor()(img), [0.5071, 0.4865, 0.4409],
                                     [0.2673, 0.2564, 0.2762]).unsqueeze(
                0)
            img_batch = input_tensor
            out = model(img_batch.cuda())
            save_path_list.append(result_img_path)
            cams = cam_extractor(out.argmax(dim=-1).detach().cpu().numpy().tolist(), out)
            cams_list = []
            for i in range(cams[0].shape[0]):
                cams_list += [cams[0][i, :, :]]
            segmaps = [to_pil_image(
                (resize(cam.unsqueeze(0), input_tensor.shape[-2:]).squeeze(0) >= thresh).to(dtype=torch.float32)) for
                cam in
                cams_list]
            for i in range(len(segmaps)):
                segmaps[i].save(save_path_list[i])
        print("%s目录下共有%d张图片！" % (origin_class_path, len(save_dirs) - last_len))
        last_len = len(save_dirs)

        #for img in tqdm(os.listdir(origin_class_path)):


    # model = model_dict['resnet32'](num_classes=100)
    # model.load_state_dict(torch.load(args.model_cam, map_location=torch.device('cpu'))['model'], strict=True)


    #for i, origion_path in enumerate(tqdm(origion_paths)):
        #ProcessFile(origion_path, save_dirs[i], model, cam_extractor, args.thresh)

    print("all datas has been moved successfully!")
