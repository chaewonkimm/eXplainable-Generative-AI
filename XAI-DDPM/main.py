from XAI_DDPM.diffusion import GaussianDiffusion
from XAI_DDPM.discriminator import Discriminator
from XAI_DDPM.train import Trainer
from XAI_DDPM.unet import Unet

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    flash_attn = True
)

diffusion = GaussianDiffusion(
    model,
    image_size = 128,
    timesteps = 1000,           # number of steps
    sampling_timesteps = 250   # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
)

params = {
    'nc' : 3,# Number of channles in the training images. For coloured images this is 3.
    'ndf' : 64, # Size of features maps in the discriminator. The depth will be multiples of this.
    }# Save step.

netD = Discriminator(params)


trainer = Trainer(
    diffusion,
    netD,
    './celeba_hq_256/',
    train_batch_size = 8,
    train_lr = 8e-5,
    train_num_steps = 1050,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay```
    amp = True,                       # turn on mixed precision
    calculate_fid = True,   # whether to calculate fid during training
)

if __name__ == '__main__':
    trainer.train()