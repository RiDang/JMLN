gg=2
set -ex
CUDA_VISIBLE_DEVICES=0,1 python train_workshop_baseline.py --dataroot dataset  --name workshop_baseline_notexture_tuning_v1 --model retrieval_workshop_baseline_tuning --dataset_mode retrieval_workshop_baseline --niter 30 --niter_decay 70 --crop_size 256 --fine_size 256 --num_threads 8 --lr 0.001 --batch_size 16  --gpu_ids 0 --continue_train --drop  --ablation $gg  --epoch_count 1   #--mode $mode # 30,70
#cd checkpoints
#mv workshop_baseline_notexture_tuning_v1 $gg
#cd ../

