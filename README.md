# Use the following command to train the model on UNC dataset 

`OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0,1,2,3 python3 main.py --batch_size 64 --num_workers 4 --optimizer AdamW --dataroot /ssd_scratch/cvit/kanishk/ --lr 2.5e-4 --weight_decay 5e-4 --image_encoder deeplabv3_plus --loss bce_l1 --dropout 0.3 --cache_type glove --epochs 100 --power_factor 0.5 --num_encoder_layers 1 --image_dim 448 --mask_dim 448 --seq_len 25 --glove_path /ssd_scratch/cvit/kanishk/glove/ --save --run_name jrm_1_cmmlf --threshold 0.4 --dataset talk2car`
