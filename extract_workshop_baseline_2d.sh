set -ex
CUDA_VISIBLE_DEVICES=0 python extract_workshop_baseline_2d.py --dataroot /home/dh/zdd/data/SketchBased/dataset --name workshop_baseline_notexture_tuning_v1 --model retrieval_workshop_baseline_tuning --dataset_mode retrieval_workshop_baseline_eval_2d --crop_size 256 --fine_size 256 #--num_threads 4 #--epoch 20 
# --epoch 30
