""" Quick n Simple Image Folder, Tarfile based DataSet

Hacked together by / Copyright 2019, Ross Wightman
"""
import torch.utils.data as data
import os
import torch
import logging
import numpy as np
from PIL import Image

from .parsers import create_parser

from .transforms import RandStainNA_Attention

_logger = logging.getLogger(__name__)


_ERROR_RETRY = 50


class ImageDataset(data.Dataset):

    def __init__(
            self,
            root,
            parser=None,
            class_map=None,
            load_bytes=False,
            transform=None,
            target_transform=None,
    ):
        if parser is None or isinstance(parser, str):
            parser = create_parser(parser or '', root=root, class_map=class_map)
        self.parser = parser
        self.load_bytes = load_bytes
        self.transform = transform
        self.target_transform = target_transform
        self._consecutive_errors = 0

    def __getitem__(self, index):
        img, target = self.parser[index]
        try:
            img = img.read() if self.load_bytes else Image.open(img).convert('RGB')
        except Exception as e:
            _logger.warning(f'Skipped sample (index {index}, file {self.parser.filename(index)}). {str(e)}')
            self._consecutive_errors += 1
            if self._consecutive_errors < _ERROR_RETRY:
                return self.__getitem__((index + 1) % len(self.parser))
            else:
                raise e
        self._consecutive_errors = 0
        if self.transform is not None:
            img = self.transform(img)
        if target is None:
            target = -1
        elif self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.parser)

    def filename(self, index, basename=False, absolute=False):
        return self.parser.filename(index, basename, absolute)

    def filenames(self, basename=False, absolute=False):
        return self.parser.filenames(basename, absolute)


class IterableImageDataset(data.IterableDataset):

    def __init__(
            self,
            root,
            parser=None,
            split='train',
            is_training=False,
            batch_size=None,
            repeats=0,
            download=False,
            transform=None,
            target_transform=None,
    ):
        assert parser is not None
        if isinstance(parser, str):
            self.parser = create_parser(
                parser, root=root, split=split, is_training=is_training,
                batch_size=batch_size, repeats=repeats, download=download)
        else:
            self.parser = parser
        self.transform = transform
        self.target_transform = target_transform
        self._consecutive_errors = 0

    def __iter__(self):
        for img, target in self.parser:
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)
            yield img, target

    def __len__(self):
        if hasattr(self.parser, '__len__'):
            return len(self.parser)
        else:
            return 0

    def filename(self, index, basename=False, absolute=False):
        assert False, 'Filename lookup by index not supported, use filenames().'

    def filenames(self, basename=False, absolute=False):
        return self.parser.filenames(basename, absolute)


class AugMixDataset(torch.utils.data.Dataset):
    """Dataset wrapper to perform AugMix or other clean/augmentation mixes"""

    def __init__(self, dataset, num_splits=2):
        self.augmentation = None
        self.normalize = None
        self.dataset = dataset
        if self.dataset.transform is not None:
            self._set_transforms(self.dataset.transform)
        self.num_splits = num_splits

    def _set_transforms(self, x):
        assert isinstance(x, (list, tuple)) and len(x) == 3, 'Expecting a tuple/list of 3 transforms'
        self.dataset.transform = x[0]
        self.augmentation = x[1]
        self.normalize = x[2]

    @property
    def transform(self):
        return self.dataset.transform

    @transform.setter
    def transform(self, x):
        self._set_transforms(x)

    def _normalize(self, x):
        return x if self.normalize is None else self.normalize(x)

    def __getitem__(self, i):
        x, y = self.dataset[i]  # all splits share the same dataset base transform
        x_list = [self._normalize(x)]  # first split only normalizes (this is the 'clean' split)
        # run the full augmentation on the remaining splits
        for _ in range(self.num_splits - 1):
            x_list.append(self._normalize(self.augmentation(x)))
        return tuple(x_list), y

    def __len__(self):
        return len(self.dataset)

#7.18添加
class CRC_ours():

    def __init__(self, root, randstainna_attention=None, transform=None,cam_name=None):
        self.root = os.path.join(root, 'train')
        class_list = os.listdir(self.root)
        data_list = []
        cam_list = []
        label_list = []
        label_num_list = []
  
    # 9.24修改，randstainna_attention从外部传入
#         color_jitter = {}
#         color_jitter['brightness'] = 0.35
#         color_jitter['contrast'] = 0.5
#         color_jitter['saturation'] = 0.1
#         color_jitter['hue'] = 0.1
#         color_jitter['p'] = 1
        
#         randstainna = {}
#         randstainna['yaml_file'] = '/root/autodl-tmp/3-RandStainNA/pytorch-image-models/norm_jitter/CRC_LAB_randomTrue_n0.yaml'
#         randstainna['std_hyper'] = -0.5
#         randstainna['probability'] = 0.5
#         randstainna['distribution'] = 'normal'
        
#         randstainna_attention = {}
#         randstainna_attention['color_jitter'] = color_jitter
#         randstainna_attention['randstainna'] = randstainna
#         randstainna_attention['fg'] = 'randstainna'
#         randstainna_attention['bg'] = 'randstainna'
        
        print(randstainna_attention)
        
        for class_ in class_list:
            class_path = os.path.join(self.root, class_)
            for img in os.listdir(class_path):
                img_path = os.path.join(class_path, img)
                #cam_dir = class_path.replace('train', cam_name, 1)
                cam_dir_name = os.path.join('train_cam',cam_name)
                cam_dir = class_path.replace('train', cam_dir_name, 1)#train刚好也是火车，也会换掉
                cam_path = os.path.join(cam_dir, img) # 同一层目录下，train是原始数据，cam是相同命名的cam数据
                data_list.append(img_path)
                cam_list.append(cam_path)
                label_num_list.append(CRC_CLASSES[class_])
                    
        self.data = data_list
        self.cam = cam_list
        self.label = label_num_list
        # self.label_list = label_list
        self.color_jitter = randstainna_attention['color_jitter']
        self.randstainna = randstainna_attention['randstainna']
        self.fg = randstainna_attention['fg']
        self.bg = randstainna_attention['bg']
        self.transform = transform
        
    def __getitem__(self, index):

        img, cam, label = self.data[index], self.cam[index], self.label[index]
        # test_img = cv2.imread(img)
        # print(test_img.shape)
        img = Image.open(img).convert('RGB')
        cam = Image.open(cam).convert('L') #黑：0，白：255
        # print(np.max(np.array(cam))) #255
        
        if self.randstainna and self.color_jitter is not None:
            # print('1:', np.array(img))
            img,avg_fg,std_fg,avg_bg,std_bg,per_fg = RandStainNA_Attention(fg=self.fg, bg=self.bg, color_jitter=self.color_jitter, randstainna=self.randstainna, seg=cam)(img)
            # print('2:', np.array(img))
        fg_para = np.concatenate((avg_fg,std_fg))
        bg_para = np.concatenate((avg_bg,std_bg))
        if self.transform is not None:
            img = self.transform(img) #目前只有一组
            img = img #.unsqueeze(0) 

        return img, label,fg_para,bg_para,per_fg

    def __len__(self):
        return len(self.data)
    
CRC_CLASSES={'ADI': 0, 'DEB': 1, 'LYM': 2, 'MUC': 3, 'MUS': 4, 'NORM': 5, 'STR': 6, 'TUM': 7}