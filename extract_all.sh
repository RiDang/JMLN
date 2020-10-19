point=75
mode=no_norm
gg=attn_cat_bala
echo $point
set -ex
CUDA_VISIBLE_DEVICES=1 python extract_workshop_baseline_2d.py --dataroot dataset --name workshop_baseline_notexture_tuning_v1 --model retrieval_workshop_baseline_tuning --dataset_mode retrieval_workshop_baseline_eval_2d --crop_size 256 --fine_size 256  --epoch $point --ablation $gg  --mode $mode #--epoch 30


set -ex
CUDA_VISIBLE_DEVICES=1 python extract_workshop_baseline_3d.py --dataroot dataset  --name workshop_baseline_notexture_tuning_v1 --model retrieval_workshop_baseline_tuning --dataset_mode retrieval_workshop_baseline_eval_3d --crop_size 256 --fine_size 256  --epoch $point --ablation $gg --mode $mode 

cd metric
python  pr_curve_shrec13.py

