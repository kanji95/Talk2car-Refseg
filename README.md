# Use the following command to train the model on UNC dataset 

```CUDA_VISIBLE_DEVICES=0,1,2,3 python3 main.py --batch_size 50 --num_workers 4 --optimizer AdamW --dataroot <dataset_path> --lr 2.5e-4 --weight_decay 5e-4 --image_encoder deeplabv3_plus --loss bce --dropout 0.3 --epochs 25 --task unc --power_factor 0.5 --num_encoder_layers 2 --image_dim 448 --mask_dim 112 --seq_len 25 --glove_path <glove embedding path> --threshold 0.40```
