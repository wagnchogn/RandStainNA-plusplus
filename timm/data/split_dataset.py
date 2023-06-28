import os
import glob
import shutil
from sklearn.model_selection import train_test_split

ori_route = '/media/wagnchogn/data_16t/NCT/NCT-CRC-HE-100K-NONORM'
new_route = '/media/wagnchogn/data/wsi_augmentation/randstiannav2-master/crc_dataset'
clss = os.listdir(ori_route)
remove_cls = ['BACK']
for cls in clss:
    if cls in remove_cls:
        continue
    files = os.listdir(os.path.join(ori_route, cls))
    x_train, x_test = train_test_split(files,test_size=0.2, random_state=42)
    for train_file in x_train:
        new_train_route = os.path.join(new_route, 'train',cls)
        if not os.path.exists(new_train_route):
            os.makedirs(new_train_route)
        train_file_path = os.path.join(ori_route, cls, train_file)
        shutil.copy(train_file_path, new_train_route)
    for test_file in x_test:
        new_test_route = os.path.join(new_route, 'val',cls)
        if not os.path.exists(new_test_route):
            os.makedirs(new_test_route)
        test_file_path = os.path.join(ori_route, cls, test_file)
        shutil.copy(test_file_path, new_test_route)
