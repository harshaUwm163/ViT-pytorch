python train.py --name birds_ft --dataset inet1k_birds --dataset_dir data/inet1k_classes/birds --model_type ViT-B_16 --pretrained_dir checkpoint/ViT-B_16.npz --eval_every 100 --num_steps 2500 --save_every 500 --train_batch_size 32 --eval_batch_size 32 
