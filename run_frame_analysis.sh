# python analyze_ViT_frames.py --name debug_thread --dataset inet1k_dogs --dataset_dir data/inet1k_classes/dogs --model_type ViT-B_16 --pretrained_dir checkpoint/ViT-B_16.npz --eval_every 100 --num_steps 2500 --save_every 500 --train_batch_size 128 --eval_batch_size 128
# python analyze_ViT_frames.py --name debug_thread --dataset inet1k_trucks --dataset_dir data/inet1k_classes/trucks --model_type ViT-B_16 --pretrained_dir checkpoint/ViT-B_16.npz --eval_every 100 --num_steps 2500 --save_every 500 --train_batch_size 128 --eval_batch_size 128
# oct 9
echo "running bash script"
# python frame_analysis_submodular.py --name inet1k_birds \
#                                     --dataset inet1k_birds \
#                                     --dataset_dir data/inet1k_classes/birds \
#                                     --model_type ViT-B_16 \
#                                     --pretrained_dir checkpoint/ViT-B_16.npz \
#                                     --eval_batch_size 128 \
#                                     --ckpt_path output/inet1k_birds-2023-10-02-22-25-22/inet1k_birds_final_ckpt.bin
# python frame_analysis_submodular.py --name inet1k_cats \
#                                     --dataset inet1k_cats \
#                                     --dataset_dir data/inet1k_classes/cats \
#                                     --model_type ViT-B_16 \
#                                     --pretrained_dir checkpoint/ViT-B_16.npz \
#                                     --eval_batch_size 128 \
#                                     --ckpt_path output/inet1k_cats-2023-10-02-22-19-15/inet1k_cats_final_ckpt.bin
# python frame_analysis_submodular.py --name inet1k_dogs \
#                                     --dataset inet1k_dogs \
#                                     --dataset_dir data/inet1k_classes/dogs \
#                                     --model_type ViT-B_16 \
#                                     --pretrained_dir checkpoint/ViT-B_16.npz \
#                                     --eval_batch_size 128 \
#                                     --ckpt_path output/inet1k_dogs-2023-09-24-21-00-17/inet1k_dogs_final_ckpt.bin
# python frame_analysis_submodular.py --name inet1k_snakes \
#                                     --dataset inet1k_snakes \
#                                     --dataset_dir data/inet1k_classes/snakes \
#                                     --model_type ViT-B_16 \
#                                     --pretrained_dir checkpoint/ViT-B_16.npz \
#                                     --eval_batch_size 128 \
#                                     --ckpt_path output/inet1k_snakes-2023-10-02-22-28-06/inet1k_snakes_final_ckpt.bin
python frame_analysis_submodular.py --name inet1k_trucks \
                                    --dataset inet1k_trucks \
                                    --dataset_dir data/inet1k_classes/trucks \
                                    --model_type ViT-B_16 \
                                    --pretrained_dir checkpoint/ViT-B_16.npz \
                                    --eval_batch_size 128 \
                                    --ckpt_path output/inet1k_trucks-2023-09-24-20-47-28/inet1k_trucks_final_ckpt.bin
echo "ended bash script"