# python vit.py --coef_est_type weiner --exp_name van_gptq_16 --wbits 16
# python vit.py --coef_est_type weiner --exp_name van_gptq_4 --wbits 4
# python vit.py --coef_est_type weiner --exp_name van_gptq_3 --wbits 3
# python vit.py --coef_est_type weiner --exp_name van_gptq_2 --wbits 2

# python vit.py --coef_est_type weiner --exp_name tff_gptq_16 --wbits 16
# python vit.py --coef_est_type weiner --exp_name tff_gptq_4 --wbits 4
# python vit.py --coef_est_type weiner --exp_name tff_gptq_3 --wbits 3
# python vit.py --coef_est_type weiner --exp_name tff_gptq_2 --wbits 2

# python vit.py --coef_est_type weiner --exp_name full_weiner_tff_gptq_16 --wbits 16
# python vit.py --coef_est_type weiner --exp_name full_weiner_tff_gptq_4 --wbits 4
# python vit.py --coef_est_type weiner --exp_name full_weiner_tff_gptq_3 --wbits 3
# python vit.py --coef_est_type weiner --exp_name full_weiner_tff_gptq_2 --wbits 2

# python vit.py --coef_est_type weiner --exp_name debug_thread --wbits 2

# python vit.py --coef_est_type weiner --exp_name fullW_tff_vitFT_16 --wbits 16 --dataset inet1k_cats --dataset_dir data/inet1k_classes/cats \
#  --ckpt_path output/olvi1_outputs/inet1k_cats-2023-10-02-22-19-15/inet1k_cats_final_ckpt.bin --parent_dir cats_vitFT 
# python vit.py --coef_est_type weiner --exp_name fullW_tff_vitFT_4 --wbits 4 --dataset inet1k_cats --dataset_dir data/inet1k_classes/cats \
#  --ckpt_path output/olvi1_outputs/inet1k_cats-2023-10-02-22-19-15/inet1k_cats_final_ckpt.bin --parent_dir cats_vitFT 
# python vit.py --coef_est_type weiner --exp_name fullW_tff_vitFT_3 --wbits 3 --dataset inet1k_cats --dataset_dir data/inet1k_classes/cats \
#  --ckpt_path output/olvi1_outputs/inet1k_cats-2023-10-02-22-19-15/inet1k_cats_final_ckpt.bin --parent_dir cats_vitFT 
# python vit.py --coef_est_type weiner --exp_name fullW_tff_vitFT_2 --wbits 2 --dataset inet1k_cats --dataset_dir data/inet1k_classes/cats \
#  --ckpt_path output/olvi1_outputs/inet1k_cats-2023-10-02-22-19-15/inet1k_cats_final_ckpt.bin --parent_dir cats_vitFT 

# python vit.py --coef_est_type weiner --exp_name gptq_vitFT_16 --wbits 16 --dataset inet1k_cats --dataset_dir data/inet1k_classes/cats \
#  --ckpt_path output/olvi1_outputs/inet1k_cats-2023-10-02-22-19-15/inet1k_cats_final_ckpt.bin --parent_dir cats_vitFT 
# python vit.py --coef_est_type weiner --exp_name gptq_vitFT_4 --wbits 4 --dataset inet1k_cats --dataset_dir data/inet1k_classes/cats \
#  --ckpt_path output/olvi1_outputs/inet1k_cats-2023-10-02-22-19-15/inet1k_cats_final_ckpt.bin --parent_dir cats_vitFT 
# python vit.py --coef_est_type weiner --exp_name gptq_vitFT_3 --wbits 3 --dataset inet1k_cats --dataset_dir data/inet1k_classes/cats \
#  --ckpt_path output/olvi1_outputs/inet1k_cats-2023-10-02-22-19-15/inet1k_cats_final_ckpt.bin --parent_dir cats_vitFT 
# python vit.py --coef_est_type weiner --exp_name gptq_vitFT_2 --wbits 2 --dataset inet1k_cats --dataset_dir data/inet1k_classes/cats \
#  --ckpt_path output/olvi1_outputs/inet1k_cats-2023-10-02-22-19-15/inet1k_cats_final_ckpt.bin --parent_dir cats_vitFT 

python vit.py --coef_est_type weiner --exp_name tff_vitFT_16 --wbits 16 --dataset inet1k_cats --dataset_dir data/inet1k_classes/cats \
 --ckpt_path output/olvi1_outputs/inet1k_cats-2023-10-02-22-19-15/inet1k_cats_final_ckpt.bin --parent_dir cats_vitFT 
python vit.py --coef_est_type weiner --exp_name tff_vitFT_4 --wbits 4 --dataset inet1k_cats --dataset_dir data/inet1k_classes/cats \
 --ckpt_path output/olvi1_outputs/inet1k_cats-2023-10-02-22-19-15/inet1k_cats_final_ckpt.bin --parent_dir cats_vitFT 
python vit.py --coef_est_type weiner --exp_name tff_vitFT_3 --wbits 3 --dataset inet1k_cats --dataset_dir data/inet1k_classes/cats \
 --ckpt_path output/olvi1_outputs/inet1k_cats-2023-10-02-22-19-15/inet1k_cats_final_ckpt.bin --parent_dir cats_vitFT 
python vit.py --coef_est_type weiner --exp_name tff_vitFT_2 --wbits 2 --dataset inet1k_cats --dataset_dir data/inet1k_classes/cats \
 --ckpt_path output/olvi1_outputs/inet1k_cats-2023-10-02-22-19-15/inet1k_cats_final_ckpt.bin --parent_dir cats_vitFT 