import cv2
import numpy as np
from skimage import color
from typing import Optional, Dict
import yaml
import os
import time

class Dict2Class(object):
      
    def __init__(
        self, 
        my_dict: Dict
    ):
        self.my_dict = my_dict  
        for key in my_dict:
            setattr(self, key, my_dict[key])
            
def get_yaml_data(yaml_file):
        file = open(yaml_file, 'r', encoding="utf-8")
        file_data = file.read()
        file.close()
        # str->dict
        data = yaml.load(file_data, Loader=yaml.FullLoader)

        return data
    
class RandStainNA(object):

    def __init__(
        self, 
        yaml_file: str,
        std_hyper: Optional[float] = 0, 
        distribution: Optional[str] = 'normal', 
        probability: Optional[float] = 1.0 ,
        is_train: bool = True 
    ):
        self.yaml_file = yaml_file
        cfg = get_yaml_data(self.yaml_file)
        c_s= cfg['color_space']
        self._channel_avgs = {
            'avg' : [cfg[c_s[0]]['avg']['mean'], cfg[c_s[1]]['avg']['mean'], cfg[c_s[2]]['avg']['mean']],
            'std' : [cfg[c_s[0]]['avg']['std'], cfg[c_s[1]]['avg']['std'], cfg[c_s[2]]['avg']['std']]
        }
        self._channel_stds = {
            'avg' : [cfg[c_s[0]]['std']['mean'], cfg[c_s[1]]['std']['mean'], cfg[c_s[2]]['std']['mean']],
            'std' : [cfg[c_s[0]]['std']['std'], cfg[c_s[1]]['std']['std'], cfg[c_s[2]]['std']['std']],
        }
        self.channel_avgs = Dict2Class(self._channel_avgs)
        self.channel_stds = Dict2Class(self._channel_stds)
        self.color_space = cfg['color_space']
        self.p = probability
        self.std_adjust = std_hyper 
        self.color_space = c_s
        self.distribution = distribution 
        self.is_train=is_train #true:training setting/false: demo setting
    
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
                                    high =self.channel_avgs.avg[idx] - 3 * self.channel_avgs.std[idx]
                                )
                tar_std = np.random.uniform(
                                    low = self.channel_avgs.avg[idx] - 3 * self.channel_avgs.std[idx],
                                    high =self.channel_avgs.avg[idx] - 3 * self.channel_avgs.std[idx]
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
        format_string += f", distribution={self.distribution}"  # 1.30添加，print期望分布
        format_string += f", p={self.p})"
        return format_string

if __name__ == '__main__':
   
    '''
    Usage1: Demo(for visualization)
    '''
    # Setting: is_train = False
    randstainna = RandStainNA(
        yaml_file = '/media/wagnchogn/data/wsi_augmentation/randstiannav2-master/crc/CRC.yaml',
        #std_hyper = -0.3,
        std_hyper=0.0,
        distribution = 'normal', 
        probability = 1.0,
        is_train = False 
    )
    print(randstainna)

    img_path_list = [
        '/media/wagnchogn/data_16t/NCT/NCT-CRC-HE-100K-NONORM/TUM/TUM-AEPINLNQ.tif',
        '/media/wagnchogn/data_16t/NCT/NCT-CRC-HE-100K-NONORM/TUM/TUM-DFGFFNEY.tif',
        '/media/wagnchogn/data_16t/NCT/NCT-CRC-HE-100K-NONORM/TUM/TUM-EWFNFSQL.tif'
        #'/media/wagnchogn/data_16t/NCT/NCT-CRC-HE-100K-NONORM/TUM/TUM-TCGA-CVATFAAT.png'
    ]
    save_dir_path = '/media/wagnchogn/data/wsi_augmentation/randstiannav2-master/crc/randstainna'
    if not os.path.exists(save_dir_path):
        os.mkdir(save_dir_path)

    t_start = time.time()
    for img_path in img_path_list:
        img = randstainna(cv2.imread(img_path))
        save_img_path = save_dir_path + '/{}'.format(img_path.split('/')[-1])
        cv2.imwrite(save_img_path,img)
    t_end = time.time()
    print(t_end-t_start)
    '''
    Usage2：torchvision.transforms (for training)
    '''
    # Setting: is_train = False
    # from torchvision import transforms
    # transforms_list = [
    #     RandStainNA(yaml_file='./CRC_LAB_randomTrue_n0.yaml', std_hyper=0, probability=1.0,
    #                           distribution='normal', is_train=True)
    # ]
    # transforms.Compose(transforms_list)
    