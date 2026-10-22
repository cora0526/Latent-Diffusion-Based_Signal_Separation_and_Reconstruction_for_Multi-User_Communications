# Latent Diffusion Based Signal Separation

使用方法:
環境:
使用python 3.11
有gpu
﻿torch==2.10.0+cu126
torchvision==0.25.0+cu126
沒有gpu
torch==2.10.0
torchvision==0.25.0
先預訓練 vae_gan encoder
python train_vae_gan.py
他會存入./save_model
在訓練 ddpm seperator
python train_ddpm.py
這樣就完成全部的預訓練
