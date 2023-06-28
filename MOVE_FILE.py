import os
import shutil
import glob
ori_route = '/home/wagnchogn/Desktop/randstiannav2/crc_dataset'
new_route = '/home/wagnchogn/Desktop/randstiannav2/analysis_dataset'
cam_files = ['cam','cam_baseline']
sel_route = '/home/wagnchogn/Desktop/randstiannav2/analysis_dataset/train'
sel_pics = os.listdir(sel_route)

if not os.path.exists(new_route):
    os.makedirs(new_route)
for cam_file in cam_files:
    cam_path = os.path.join(ori_route,cam_file)
    for sel_file in sel_pics:
        cls = sel_file.split('-')[0]
        ori_pic = os.path.join(cam_path,cls,sel_file)
        new_path = os.path.join(new_route,cam_file+'_'+sel_file)
        shutil.copy(ori_pic,new_path)