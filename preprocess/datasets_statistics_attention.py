import os
import cv2
import numpy as np
import time
import argparse
import yaml
import math
import json
import random
import copy
from skimage import color
from fitter import Fitter
import pandas as pd
from PIL import Image
parser = argparse.ArgumentParser(description='norm&jitter dataset lab statistics')
parser.add_argument('--data-dir', type=str, metavar='DIR', default='/home/xdjf/Desktop/crc_dataset/train',
                    help='path to dataset(ImageFolder form)') 
parser.add_argument('--save-dir', type=str,metavar='DIR', default='/home/xdjf/Desktop/crc_dataset/crc_yaml_v1',
                    help='path to save dataset')
parser.add_argument('--cam-dir', type=str, metavar='DIR', default='/home/xdjf/Desktop/crc_dataset/train_cam',
                    help='path to save dataset')
parser.add_argument('--dataset-name', type=str, default='', metavar='DIR',
                    help='dataset output_name')
parser.add_argument('--methods', type=str, default='Reinhard',
                    help='colornorm_methods')
parser.add_argument('--color-space', type=str, default='LAB', choices=['LAB', 'HED', 'HSV'],
                    help='dataset statistics color space')
parser.add_argument('--random', action='store_true', default=True,
                    help='random shuffle sample')
parser.add_argument('--n', type=int, default=0, metavar='DIR',
                    help='datasets statistics sample n image each class(0:all)')


def _parse_args():
    args = parser.parse_args()

    return args


def getavgstd(image,seg_np):
    avg = []
    std = []
    #a = image[:, :, 0]
    b = seg_np[:,:,0]
    image_avg_l = np.mean((image[:, :, 0])[b==1])
    image_std_l = np.std((image[:, :, 0])[b==1])
    image_avg_a = np.mean((image[:, :, 1])[b==1])
    image_std_a = np.std((image[:, :, 1])[b==1])
    image_avg_b = np.mean((image[:, :, 2])[b==1])
    image_std_b = np.std((image[:, :, 2])[b==1])


    avg.append(image_avg_l)
    avg.append(image_avg_a)
    avg.append(image_avg_b)
    std.append(image_std_l)
    std.append(image_std_a)
    std.append(image_std_b)
    return (avg, std)


if __name__ == '__main__':

    args = _parse_args()
    img_bg_fg_list = ['bg','fg']
    #img_bg_fg_list = ['fg']
    for img_bg_fg in img_bg_fg_list:
        path_dataset = args.data_dir
        path_cam_route = args.cam_dir
        modes = os.listdir(path_cam_route)

        for mode in modes:
            path_cam_dataset = os.path.join(path_cam_route,mode)
            labL_avg_List = []
            labA_avg_List = []
            labB_avg_List = []
            labL_std_List = []
            labA_std_List = []
            labB_std_List = []

            t1 = time.time()
            i = 0

            for class_dir in os.listdir(path_dataset):
                path_class = os.path.join(path_dataset, class_dir)
                cam_path_class = os.path.join(path_cam_dataset, class_dir)
                #print(path_class)

                path_class_list = os.listdir(path_class)
                if args.random:
                    random.shuffle(path_class_list)

                for image in path_class_list:
                    if args.n == 0:  # n=0: all images each class
                        pass
                    elif i < args.n:
                        i += 1
                    else:
                        i = 0
                        break
                    path_img = os.path.join(path_class, image)
                    path_cam = os.path.join(cam_path_class, image)
                    img = cv2.imread(path_img)
                    if not os.path.exists(path_cam):
                        print(path_cam)
                    cam = cv2.imread(path_cam)
                    cam = cv2.cvtColor(cam, cv2.COLOR_BGR2GRAY)
                    seg_np = np.array(cam)[:, :, np.newaxis] / 255.0

                    if img_bg_fg == 'fg':
                        seg_np_final = seg_np
                    elif img_bg_fg == 'bg':
                        seg_np_final = 1 - seg_np
                    else:
                        print('error_get_avg_std',path_img)
                    try: #debug
                        if args.color_space == 'LAB':
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

                        elif args.color_space == 'HED':
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            img = color.rgb2hed(img)

                        elif args.color_space == 'HSV':
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

                        else:
                            print('wrong color space: {}!!'.format(args.color_space))

                        img_avg, img_std = getavgstd(img,seg_np_final)
                    except:
                        continue
                        print(path_img)
                    if (math.isnan(img_avg[0])) or (math.isnan(img_avg[1])) or (math.isnan(img_avg[2])) or (math.isnan(img_std[0])) or (math.isnan(img_std[1])) or (math.isnan(img_std[2])):
                        continue

                    labL_avg_List.append(img_avg[0])
                    labA_avg_List.append(img_avg[1])
                    labB_avg_List.append(img_avg[2])
                    labL_std_List.append(img_std[0])
                    labA_std_List.append(img_std[1])
                    labB_std_List.append(img_std[2])
            t2 = time.time()
            print(t2 - t1)
            save_path = os.path.join(args.save_dir,mode,img_bg_fg)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            save_excel_path = os.path.join(args.save_dir,mode)
            df = pd.DataFrame(data={'labA_avg': labL_avg_List, 'labA_std': labL_std_List, 'labB_avg': labB_avg_List,
                                    'labB_std': labB_std_List, 'labL_avg': labA_avg_List, 'labL_std': labA_std_List})
            df.to_excel(os.path.join(save_excel_path, 'CRC'+'_'+img_bg_fg + '.xlsx'), index=False)

            l_avg_mean = np.mean(labL_avg_List).item()
            l_avg_std = np.std(labL_avg_List).item()
            l_std_mean = np.mean(labL_std_List).item()
            l_std_std = np.std(labL_std_List).item()
            a_avg_mean = np.mean(labA_avg_List).item()
            a_avg_std = np.std(labA_avg_List).item()
            a_std_mean = np.mean(labA_std_List).item()
            a_std_std = np.std(labA_std_List).item()
            b_avg_mean = np.mean(labB_avg_List).item()
            b_avg_std = np.std(labB_avg_List).item()
            b_std_mean = np.mean(labB_std_List).item()
            b_std_std = np.std(labB_std_List).item()

            std_avg_list = [labL_avg_List, labL_std_List, labA_avg_List, labA_std_List, labB_avg_List, labB_std_List]
            distribution = []
            for std_avg in std_avg_list:
                f = Fitter(std_avg, distributions=['norm', 'laplace'])
                f.fit()
                distribution.append(list(f.get_best(method='sumsquare_error').keys())[0])

            yaml_dict_lab = {
                'random': args.random,
                'n_each_class': args.n,
                'color_space': args.color_space,
                'methods': args.methods,
                '{}'.format(args.color_space[0]): {  # lab-L/hed-H
                    'avg': {
                        'mean': round(l_avg_mean, 3),
                        'std': round(l_avg_std, 3),
                        'distribution': distribution[0]
                    },
                    'std': {
                        'mean': round(l_std_mean, 3),
                        'std': round(l_std_std, 3),
                        'distribution': distribution[1]
                    }
                },
                '{}'.format(args.color_space[1]): {  # lab-A/hed-E
                    'avg': {
                        'mean': round(a_avg_mean, 3),
                        'std': round(a_avg_std, 3),
                        'distribution': distribution[2]
                    },
                    'std': {
                        'mean': round(a_std_mean, 3),
                        'std': round(a_std_std, 3),
                        'distribution': distribution[3]
                    }
                },
                '{}'.format(args.color_space[2]): {  # lab-B/hed-D
                    'avg': {
                        'mean': round(b_avg_mean, 3),
                        'std': round(b_avg_std, 3),
                        'distribution': distribution[4]
                    },
                    'std': {
                        'mean': round(b_std_mean, 3),
                        'std': round(b_std_std, 3),
                        'distribution': distribution[5]
                    }
                }
            }

            yaml_save_path = '{}/{}.yaml'.format(
                save_path,
                args.dataset_name if args.dataset_name != '' else 'dataset_{}_random{}_n{}'.format(args.color_space,
                                                                                                   args.random, args.n))
            with open(yaml_save_path, 'w') as f:
                yaml.dump(yaml_dict_lab, f)
                print('The dataset lab statistics has been saved in {}'.format(yaml_save_path))