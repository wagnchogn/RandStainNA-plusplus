import cv2
import numpy as np
from PIL import Image
from skimage import color
from typing import Optional, Dict
import yaml
from torchvision.transforms import Compose, transforms
import random

# ToDo: add documents

class Dict2Class(object):
    # ToDo: Wrap into RandStainNA
    def __init__(
        self, 
        my_dict: Dict
    ):
        self.my_dict = my_dict  
        for key in my_dict:
            setattr(self, key, my_dict[key])
            
def get_yaml_data(yaml_file):
    # ToDo: Wrap into RandStainNA
        file = open(yaml_file, 'r', encoding="utf-8")
        file_data = file.read()
        file.close()
        # str->dict
        data = yaml.load(file_data, Loader=yaml.FullLoader)

        return data
    
class RandStainNA(object):
    # ToDo: support downloading yaml file from online if the path is not provided.
    def __init__(
        self, 
        yaml_file: str, 
        std_hyper: Optional[float] = 0, 
        distribution: Optional[str] = 'normal', 
        probability: Optional[float] = 1.0 ,
        is_train: Optional[bool] = True, 
    ):
        
        #true:training setting/false: demo setting
        
        assert distribution in ['normal','laplace','uniform'], 'Unsupported distribution style {}.'.format(distribution)
        
        self.yaml_file = yaml_file
        cfg = get_yaml_data(self.yaml_file)
        c_s= cfg['color_space']
        color_space = c_s
        #x = cfg['A']['std']['mean']
        ### 原版
        '''
        self._channel_avgs = {
            'avg': [cfg[c_s[0]]['avg']['mean'], cfg[c_s[1]]['avg']['mean'], cfg[c_s[2]]['avg']['mean']],
            'std': [cfg[c_s[0]]['avg']['std'], cfg[c_s[1]]['avg']['std'], cfg[c_s[2]]['avg']['std']]
        }
        self._channel_stds = {
            'avg': [cfg[c_s[0]]['std']['mean'], cfg[c_s[1]]['std']['mean'], cfg[c_s[2]]['std']['mean']],
            'std': [cfg[c_s[0]]['std']['std'], cfg[c_s[1]]['std']['std'], cfg[c_s[2]]['std']['std']],
        }
        '''
        new_probability = probability
        if c_s == 'LAB' or c_s == 'HSV' or c_s == 'HED':
            self._channel_avgs = {
                'avg' : [cfg[c_s[0]]['avg']['mean'], cfg[c_s[1]]['avg']['mean'], cfg[c_s[2]]['avg']['mean']],
                'std' : [cfg[c_s[0]]['avg']['std'], cfg[c_s[1]]['avg']['std'], cfg[c_s[2]]['avg']['std']]
            }
            self._channel_stds = {
                'avg' : [cfg[c_s[0]]['std']['mean'], cfg[c_s[1]]['std']['mean'], cfg[c_s[2]]['std']['mean']],
                'std' : [cfg[c_s[0]]['std']['std'], cfg[c_s[1]]['std']['std'], cfg[c_s[2]]['std']['std']],
            }

        elif c_s == 'Random':
            color_spaces = ['LAB', 'HED', 'HSV']
            color_space = random.choices(color_spaces,probability)[0]
            if color_space == 'LAB':
                self._channel_avgs = {
                    'avg': [cfg['A']['avg']['mean'], cfg['B']['avg']['mean'], cfg['L']['avg']['mean']],
                    'std': [cfg['A']['avg']['std'], cfg['B']['avg']['std'], cfg['L']['avg']['std']]
                }
                self._channel_stds = {
                    'avg': [cfg['A']['std']['mean'], cfg['B']['std']['mean'], cfg['L']['std']['mean']],
                    'std': [cfg['A']['std']['std'], cfg['B']['std']['std'], cfg['L']['std']['std']],
                }
            if color_space == 'HED':
                self._channel_avgs = {
                    'avg': [cfg['D']['avg']['mean'], cfg['E']['avg']['mean'], cfg['H']['avg']['mean']],
                   'std': [cfg['D']['avg']['std'], cfg['E']['avg']['std'], cfg['H']['avg']['std']]
                }
                self._channel_stds = {
                    'avg': [cfg['D']['std']['mean'], cfg['E']['std']['mean'], cfg['H']['std']['mean']],
                   'std': [cfg['D']['std']['std'], cfg['E']['std']['std'], cfg['H']['std']['std']],
                }
            if color_space == 'HSV':
                self._channel_avgs = {
                    'avg': [cfg['h']['avg']['mean'], cfg['S']['avg']['mean'], cfg['V']['avg']['mean']],
                  'std': [cfg['h']['avg']['std'], cfg['S']['avg']['std'], cfg['V']['avg']['std']]
                }
                self._channel_stds = {
                    'avg': [cfg['h']['std']['mean'], cfg['S']['std']['mean'], cfg['V']['std']['mean']],
                  'std': [cfg['h']['std']['std'], cfg['S']['std']['std'], cfg['V']['std']['std']],
                }

            new_probability = 1
        self.channel_avgs = Dict2Class(self._channel_avgs)
        self.channel_stds = Dict2Class(self._channel_stds)

        self.color_space = cfg['color_space']
        self.p = new_probability
        self.std_adjust = std_hyper
        self.color_space = color_space
        self.distribution = distribution
        self.is_train = is_train
    def _getavgstd(
        self, 
        image: np.ndarray,
        isReturnNumpy: Optional[bool] = True
    ):

        avgs = []
        stds = []

        num_of_channel = image.shape[2]
        for idx in range(num_of_channel):
            avgs.append(np.mean(image[:, :, idx]))
            stds.append(np.std(image[:, :, idx]))
        
        if isReturnNumpy:
            return (np.array(avgs), np.array(stds))
        else:
            return (avgs,stds)

    def _normalize(
        self, 
        img: np.ndarray, 
        img_avgs: np.ndarray, 
        img_stds: np.ndarray, 
        tar_avgs: np.ndarray, 
        tar_stds: np.ndarray,
    ) -> np.ndarray :
         
        img_stds = np.clip(img_stds, 0.0001, 255)
        img = (img - img_avgs) * (tar_stds / img_stds) + tar_avgs

        if self.color_space in ["LAB","HSV"]: 
            img = np.clip(img, 0, 255).astype(np.uint8)

        return img

    def augment(self, img): 
        #img:is_train:false——>np.array()(cv2.imread()) #BGR
        #img:is_train:True——>PIL.Image #RGB
        
        if self.is_train == False:
            image = img
        else :
            image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        num_of_channel = image.shape[2]  

        # color space transfer
        if self.color_space == 'LAB':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)  
        elif self.color_space == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  
        elif self.color_space == 'HED':  
            image = color.rgb2hed(
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            )  

        std_adjust = self.std_adjust

        # virtual template generation
        tar_avgs = []
        tar_stds = []
        if self.distribution == 'uniform':
            
            # three-sigma rule for uniform distribution
            for idx in range(num_of_channel):
                
                tar_avg = np.random.uniform(
                                    low = self.channel_avgs.avg[idx] - 3 * self.channel_avgs.std[idx],
                                    high =self.channel_avgs.avg[idx] - 3 * self.channel_avgs.std[idx],
                                )
                tar_std = np.random.uniform(
                                    low = self.channel_avgs.avg[idx] - 3 * self.channel_avgs.std[idx],
                                    high =self.channel_avgs.avg[idx] - 3 * self.channel_avgs.std[idx],
                                )
                
                tar_avgs.append(tar_avg)
                tar_stds.append(tar_std)
        else:  
            if self.distribution == 'normal':
                np_distribution = np.random.normal
            elif self.distribution == 'laplace':
                np_distribution = np.random.laplace

            for idx in range(num_of_channel):
                tar_avg = np_distribution(
                            loc=self.channel_avgs.avg[idx], 
                            scale=self.channel_avgs.std[idx] * (1 + std_adjust)
                        )

                tar_std = np_distribution(
                            loc=self.channel_stds.avg[idx], 
                            scale=self.channel_stds.std[idx] * (1 + std_adjust)
                        )
                tar_avgs.append(tar_avg)
                tar_stds.append(tar_std)
                    
        tar_avgs = np.array(tar_avgs)
        tar_stds = np.array(tar_stds)

        img_avgs, img_stds = self._getavgstd(image)

        image = self._normalize(
            img = image, 
            img_avgs = img_avgs,
            img_stds = img_stds,
            tar_avgs = tar_avgs,
            tar_stds = tar_stds,
        )

        if self.color_space == 'LAB':
            image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)
        elif self.color_space == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        elif self.color_space == 'HED':
            nimg = color.hed2rgb(image)
            imin = nimg.min()
            imax = nimg.max()
            rsimg = (255 * (nimg - imin) / (imax - imin)).astype('uint8')  # rescale to [0,255]

            image = cv2.cvtColor(rsimg, cv2.COLOR_RGB2BGR)

        if self.is_train == True:

            return Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
        else:
            return image
    
    def __call__( self, img ) : 
        if np.random.rand(1) < self.p:
            return self.augment(img)
        else:
            return img
    
    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        format_string += f"methods=Reinhard"
        format_string += f", colorspace={self.color_space}"
        format_string += f", mean={self._channel_avgs}"
        format_string += f", std={self._channel_stds}"
        format_string += f", std_adjust={self.std_adjust}"
        format_string += f", distribution={self.distribution}"  
        format_string += f", p={self.p})"
        return format_string
    
class HSVJitter(object):
    def __init__(self, brightness=0.0, contrast=0.0, saturation=0.0, hue=0.0, p=1.0):
        self.brightness=brightness
        self.contrast=contrast
        self.saturation=saturation
        self.hue=hue
        self.p=p
        self.colorJitter=transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
    
    def __call__(self, img):
        if np.random.rand(1) < self.p: #2.13加入概率
            img_process = self.colorJitter(img)
            return img_process
        else:
            return img
    
    def __repr__(self):
        format_string = "("
        format_string += self.colorJitter.__repr__()
        format_string += f", p={self.p})"
        return format_string