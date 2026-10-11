from dataclasses import dataclass

# @dataclass
# class Config:
#     seed: int = 42
#     device: str = "cuda"

#     eurosat_root: str = r"D:/Python/SemCom/saved_data/EuroSAT/EuroSAT"
#     deepglobe_root: str = r"D:/Python/SemCom/saved_data/deepglobe_64/train"
#     save_root: str = "./save_data/VAE_GAN_from_scratch"
#     model_root: str = "./save_model"

#     epochs: int = 150
#     batch_size: int = 128
#     num_workers: int = 4
#     max_images_per_domain: int = 50000
#     validation_ratio: float = 0.05

#     z_dim: int = 256
#     label_dim: int = 64
#     num_classes: int = 2
#     critic_feature_dim: int = 512

#     lr_encoder: float = 1e-4
#     lr_generator: float = 1e-4
#     lr_critic: float = 5e-5
#     n_critic: int = 3
#     grad_clip: float = 5.0

#     beta_max: float = 1e-4
#     kl_warmup_epochs: int = 30

#     lambda_pixel: float = 1.0
#     lambda_feature: float = 0.1
#     lambda_recon_adv: float = 1e-3
#     lambda_random_adv: float = 1e-3
#     lambda_gp: float = 10.0

#     save_every: int = 5

"""Configuration: adjust paths and training settings here. ddpm"""
seed = 42
z_dim = 256
label_dim = 64
num_classes = 2
epochs = 100
steps_per_epoch = 1000
batch_size = 32
num_workers = 4
learning_rate = 2e-4
grad_clip = 1.0
diffusion_steps = 400
lambda_x0 = 0.2
clean_epochs = 20
min_train_snr_db = 10.0
max_train_snr_db = 30.0
validation_snr_db = 20.0
validate_every = 5
fixed_count = 16

data_root_e = r"D:/Python/SemCom/saved_data/EuroSAT/EuroSAT"
data_root_d = r"D:/Python/SemCom/saved_data/deepglobe_64/train"
joint_checkpoint_path = "./save_model/vae_gan_from_scratch_best.pth"
save_root = "./save_data/DDPM_VAEGAN_using_VAE_recipe"
model_root = "./save_model"