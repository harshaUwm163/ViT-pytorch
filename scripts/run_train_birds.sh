# # fine-tuning
# python train.py --name inet1k_birds --dataset inet1k_birds --dataset_dir data/inet1k_classes/birds --model_type ViT-B_16 --pretrained_dir checkpoint/ViT-B_16.npz --eval_every 100 --num_steps 2500 --save_every 500 --train_batch_size 128 --eval_batch_size 128
# linear-probing
python train.py --name inet1k_birds --dataset inet1k_birds --dataset_dir data/inet1k_classes/birds --model_type ViT-B_16 --pretrained_dir checkpoint/ViT-B_16_ft.npz --eval_every 500 --num_steps 2500 --save_every 500 --train_batch_size 128 --eval_batch_size 128 --linProbe