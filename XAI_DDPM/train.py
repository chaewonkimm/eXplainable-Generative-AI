from torch.utils.data import DataLoader
from torch.optim import Adam
from accelerate import Accelerator
import torch
from pathlib import Path
from functools import partial
from multiprocessing import cpu_count
from torch.cuda.amp import autocast
from torchvision import transforms as T, utils

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from tqdm.auto import tqdm
from ema_pytorch import EMA

from accelerate import Accelerator

from denoising_diffusion_pytorch.attend import Attend
from denoising_diffusion_pytorch.fid_evaluation import FIDEvaluation

from denoising_diffusion_pytorch.version import __version__

from XAI_DDPM.unet import Unet
from XAI_DDPM.diffusion import GaussianDiffusion
from XAI_DDPM.discriminator import Discriminator
from XAI_DDPM.dataset import Dataset
from XAI_DDPM.utils import *

import wandb


run = wandb.init(project="ddpm + discriminator")

wandb.config = {
  "diffsuion timesteps": 1000,
  "diffusion sampling_timesteps":250,
  "train_batch_size" : 8,
  "train_lr" : 8e-5,
  "train_num_steps": 500000,
  "batch_size": 8
}

def weights_init(w):
    """
    Initializes the weights of the layer, w.
    """
    classname = w.__class__.__name__
    if classname.find('conv') != -1:
        nn.init.normal_(w.weight.data, 0.0, 0.02)
    elif classname.find('bn') != -1:
        nn.init.normal_(w.weight.data, 1.0, 0.02)
        nn.init.constant_(w.bias.data, 0)


class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        discriminator,
        folder,  # 데이터 루트
        *,
        train_batch_size=16,  # 훈련 배치 크기
        gradient_accumulate_every=1,  # 가중치 업데이트 빈도
        augment_horizontal_flip=True,  # 데이터 증강 중 수평 뒤집기 사용 여부
        train_lr=1e-4,  # 훈련 학습률
        train_num_steps=100000,  # 총 훈련 스텝 수
        ema_update_every=10,  # Exponential Moving Average 업데이트 빈도
        ema_decay=0.995,  # EMA 데케이율
        adam_betas=(0.9, 0.99),  # Adam 옵티마이저 베타 값
        save_and_sample_every=1000,  # 주기적으로 저장 및 샘플링하는 빈도
        num_samples=25,  # 생성할 샘플 수
        results_folder='./results',  # 결과를 저장할 폴더
        amp=False,  # Automatic Mixed Precision (AMP) 사용 여부
        mixed_precision_type='fp16',  # 혼합 정밀도 유형
        split_batches=True,  # 배치 분할 사용 여부
        convert_image_to=None,  # 이미지 변환 설정
        calculate_fid=True,  # FID 계산 사용 여부
        inception_block_idx=2048,  # Inception block 인덱스
        max_grad_norm=1.,  # 최대 그래디언트 노름
        num_fid_samples=50000,  # FID 계산에 사용할 샘플 수
        save_best_and_latest_only=False,  # 최고 FID 및 최신 모델만 저장 여부
        discriminator_lr=0.0002,
        discriminator_adam_betas = (0.5,0.999)

    ):
        super().__init__()

        # 가속기
        self.accelerator = Accelerator(
            split_batches=split_batches,
            mixed_precision=mixed_precision_type if amp else 'no'
        )

        # 모델
        self.model = diffusion_model
        self.channels = diffusion_model.channels
        is_ddim_sampling = diffusion_model.is_ddim_sampling

        self.discriminator = discriminator
        discriminator.apply(weights_init)

        # 채널에 따른 기본 convert_image_to 설정
        if not exists(convert_image_to):
            convert_image_to = {1: 'L', 3: 'RGB', 4: 'RGBA'}.get(self.channels)

        # 샘플링 및 훈련 하이퍼파라미터
        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        assert (train_batch_size * gradient_accumulate_every) >= 16, f'your effective batch size (train_batch_size x gradient_accumulate_every) should be at least 16 or above'

        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size

        self.max_grad_norm = max_grad_norm

        # 데이터셋 및 데이터로더
        self.ds = Dataset(folder, self.image_size, augment_horizontal_flip=augment_horizontal_flip, convert_image_to=convert_image_to)

        assert len(self.ds) >= 100, 'you should have at least 100 images in your folder. at least 10k images recommended'

        dl = DataLoader(self.ds, batch_size=train_batch_size, shuffle=True, pin_memory=True, num_workers=cpu_count())
        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)

        # 옵티마이저
        self.opt = Adam(diffusion_model.parameters(), lr=train_lr, betas=adam_betas)
        self.discriminator_opt = Adam(discriminator.parameters(), lr=discriminator_lr, betas=discriminator_adam_betas)

        # 주기적으로 폴더에 결과 기록
        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta=ema_decay, update_every=ema_update_every)
            self.ema.to(self.device)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True)

        # 스텝 카운터 상태
        self.step = 0

        # 가속기를 사용하여 모델, 데이터로더, 옵티마이저 준비
        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

        # FID 스코어 계산
        self.calculate_fid = calculate_fid and self.accelerator.is_main_process

        if self.calculate_fid:
            if not is_ddim_sampling:
                self.accelerator.print(
                    "WARNING: Robust FID computation requires a lot of generated samples and can therefore be very time-consuming."
                    "Consider using DDIM sampling to save time."
                )
            self.fid_scorer = FIDEvaluation(
                batch_size=self.batch_size,
                dl=self.dl,
                sampler=self.ema.ema_model,
                channels=self.channels,
                accelerator=self.accelerator,
                stats_dir=results_folder,
                device=self.device,
                num_fid_samples=num_fid_samples,
                inception_block_idx=inception_block_idx
            )

        if save_best_and_latest_only:
            assert calculate_fid, "`calculate_fid` must be True to provide a means for model evaluation for `save_best_and_latest_only`."
            self.best_fid = 1e10  # 무한대

        self.save_best_and_latest_only = save_best_and_latest_only

    @property
    def device(self):
        return self.accelerator.device

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
            'version': __version__
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        if 'version' in data:
            print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])
    
    def compute_discriminator_loss(self,discriminator, real_images, generated_images,device):
        real_labels = torch.ones((len(real_images), 1), device=device)
        fake_labels = torch.zeros((len(generated_images), 1), device=device)
        self.discriminator = self.discriminator.to(device)

        # 진짜 이미지에 대한 손실
        #with torch.inference_mode():
        real_loss = F.binary_cross_entropy_with_logits(discriminator(real_images), real_labels)
            # 생성된 이미지에 대한 손실
        fake_loss = F.binary_cross_entropy_with_logits(discriminator(generated_images), fake_labels)

        # 전체 디스크리미네이터 손실
        total_loss = real_loss + fake_loss
        return total_loss

    
    def update_discriminator(self, real_images,device):
        # 생성된 이미지 샘플링
        generated_images = self.ema.ema_model.sample(batch_size=len(real_images))
        # 진짜 이미지와 생성된 이미지에 대한 디스크리미네이터 손실 계산
        cloned_generated_images = generated_images.clone()
        discriminator_loss = self.compute_discriminator_loss(self.discriminator, real_images, cloned_generated_images,device)
        # 디스크리미네이터 업데이트
        #self.discriminator_opt.step()
        #self.discriminator_opt.zero_grad()
        return discriminator_loss

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        with tqdm(initial=self.step, total=self.train_num_steps, disable=not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:

                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    data = next(self.dl).to(device)

                    with self.accelerator.autocast():
                        loss = self.model(data)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)

                    discriminator_loss = self.update_discriminator(data, device)

                # Discriminator의 기울기를 초기화
                self.discriminator_opt.zero_grad()

                # 모델 및 Discriminator의 기울기 합산
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                #loss.requires_grad_(True)

                self.accelerator.backward(discriminator_loss)

                # Optimizer 업데이트
                self.opt.step()
                self.opt.zero_grad()

                # Discriminator Optimizer 업데이트
                self.discriminator_opt.step()
                self.discriminator_opt.zero_grad()

                pbar.set_description(f'DDPM loss: {total_loss:.4f} / Discriminator loss: {discriminator_loss:.4f}')
                wandb.log({"DDPM_loss": total_loss,"Discriminator loss":discriminator_loss}, step=self.step)

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.update()

                    if self.step != 0 and divisible_by(self.step, self.save_and_sample_every):
                        self.ema.ema_model.eval()

                        with torch.inference_mode():
                            milestone = self.step // self.save_and_sample_every
                            batches = num_to_groups(self.num_samples, self.batch_size)
                            all_images_list = list(map(lambda n: self.ema.ema_model.sample(batch_size=n), batches))

                        all_images = torch.cat(all_images_list, dim=0)

                        utils.save_image(all_images, str(self.results_folder / f'sample-{milestone}.png'),
                                        nrow=int(math.sqrt(self.num_samples)))

                        # 이부분에 대해 Discriminator 달면 될듯

                        # FID를 계산할지 여부
                        if self.calculate_fid:
                            fid_score = self.fid_scorer.fid_score()
                            accelerator.print(f'fid_score: {fid_score}')
                            wandb.log({"fid_score": fid_score}, step=step)
                        if self.save_best_and_latest_only:
                            if self.best_fid > fid_score:
                                self.best_fid = fid_score
                                self.save("best")
                            self.save("latest")
                        else:
                            self.save(milestone)

                pbar.update(1)

        accelerator.print('training complete')