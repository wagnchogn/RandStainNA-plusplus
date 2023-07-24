model_name='efficientnetb0'
dataset_path='../crc_dataset'
b=128
workers=15
lr=0.1
num_classes=8
train_path='../randstaina++/train.py'
nj_random_path='../crc_dataset/crc_yaml'

python $train_path $dataset_path \
--model $model_name \
--num-classes $num_classes --dataset 'torch/randstainna_cv' \
--color-jitter 0.35 0.5 0 0 --hflip 0.5 --vflip 0.5 --morphology --nj-config $nj_random_path --nj-stdhyper -0.5 --nj-distribution normal --nj-p 0.3 0.3 0.4 \
--randstainna-attention randstainna randstainna \
--epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
--opt sgd --weight-decay 1e-4 --momentum 0.9 \
--sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
--experiment randstainna-attention-fg-bg-random-normal-std-0.5-p0.3-0.3-0.4_M_BC_$model_name \
-j $workers --no-prefetcher --pin-mem \
--cam_name efficientnet \
--ddsgd \
--seed 97 --native-amp

