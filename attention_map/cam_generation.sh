dataset='/home/xdjf/Desktop/randstiannav2/crc_dataset/train'
result='/home/xdjf/Desktop/randstiannav2/crc_dataset/cam'
model='/home/xdjf/Desktop/randstiannav2/randstiannav2-master/scripts_CRC/output_back/train/baseline_M_BC_labhsvstrong0.5_resnet18/best.pth'

python cam_generation.py --dataset $dataset --result $result --model-cam $model --thresh 0.5
python cam_generation_single_pic.py --dataset $dataset --result $result --model-cam $model --thresh 0.5
