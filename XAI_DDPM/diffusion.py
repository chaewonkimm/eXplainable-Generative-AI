import torch
from torch import nn
import torch.nn.functional as F
from random import random
from functools import partial
from collections import namedtuple
from torch.cuda.amp import autocast
from torchvision import transforms as T, utils
from tqdm.auto import tqdm

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange
from ema_pytorch import EMA

from denoising_diffusion_pytorch.attend import Attend
from denoising_diffusion_pytorch.fid_evaluation import FIDEvaluation

from denoising_diffusion_pytorch.version import __version__
from XAI_DDPM.unet import Unet
from XAI_DDPM.utils import *


# constants

ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start']) # 튜플 정의

# gaussian diffusion trainer class

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps): #beta scheduler
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008): #beta scheduler
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def sigmoid_beta_schedule(timesteps, start = -3, end = 3, tau = 1, clamp_min = 1e-5): #beta scheduler
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model, # Gaussian Diffusion 모델을 정의하는데 사용되는 신경망 모델
        *,
        image_size, #이미지의 크기
        timesteps=1000, #확산 프로세스에 사용되는 총 시간 단계 수
        sampling_timesteps=None, #샘플링하는 데 사용되는 시간 단계 수
        objective='pred_v', #모델의 목적을 지정하는 문자열 'pred_v', 'pred_x0', 'pred_noise' 중 하나
        #'pred_v' (Predict v): 모델은 이미지의 시작 부분에서 특정한 매개변수 v를 예측
        # 'pred_x0' (Predict x0): 모델은 이미지의 시작 부분 x0을 예측
        # 'pred_noise' (Predict noise): 모델은 확산 과정에서 각 시간 단계에서의 노이즈를 예측
        beta_schedule='sigmoid', #베타 값이 변하는 스케줄을 지정하는 문자열 'linear', 'cosine', 'sigmoid' 중 하나
        schedule_fn_kwargs=dict(), #베타 스케줄 함수에 전달되는 추가 키워드 인수
        ddim_sampling_eta=0., #ddim 샘플링에 사용되는 eta 값
        # sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
        # eta는 다음 시간 단계에서의 alpha 값과 현재 시간 단계에서의 alpha 값의 비율을 사용하여 계산된 값
        auto_normalize=True, #데이터를 자동으로 정규화할지 여부를 나타내는 불리언 값
        offset_noise_strength=0.,  #오프셋 노이즈 강도입니다. 논문에 따르면 0.1이 이상적이라고 주장 # https://www.crosslabs.org/blog/diffusion-with-offset-noise
        min_snr_loss_weight=False, #최소 SNR 손실 가중치를 사용할지 여부를 나타내는 불리언 값 # https://arxiv.org/abs/2303.09556
        min_snr_gamma=5 #최소 SNR 손실 가중치에 사용되는 감마 값
    ):
        super().__init__()
        assert not (type(self) == GaussianDiffusion and model.channels != model.out_dim)
        assert not model.random_or_learned_sinusoidal_cond

        # 모델 초기화
        self.model = model

        # 모델 관련 파라미터 설정
        self.channels = self.model.channels
        self.self_condition = self.model.self_condition

        # 이미지 크기와 목적에 대한 설정
        self.image_size = image_size
        self.objective = objective

        # 목적 함수 유형 확인
        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be pred_noise, pred_x0, or pred_v'

        # Beta 스케줄 설정
        if beta_schedule == 'linear':
            beta_schedule_fn = linear_beta_schedule
        elif beta_schedule == 'cosine':
            beta_schedule_fn = cosine_beta_schedule
        elif beta_schedule == 'sigmoid':
            beta_schedule_fn = sigmoid_beta_schedule
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        # Beta 값 계산
        betas = beta_schedule_fn(timesteps, **schedule_fn_kwargs)
        #beta_schedule_fn은 주어진 timesteps에 대한 베타 스케줄을 생성하는 함수
        #이 함수는 시간에 따라 변하는 betas를 만듬
        #시간이 흐를수록 베타 값이 어떻게 변하는지는 schedule_fn_kwargs에 따라 결정
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        #각 시간 단계에 대한 누적 알파 값(cumulative product)을 계산
        #이 값은 확산 분포에서 사용

        #torch.cumprod 함수는 주어진 텐서의 누적 곱을 계산
        #누적 곱은 각 원소가 이전 원소들의 곱으로 누적되는 과정을 나타냄
        # # [2, 3, 4]          # 원본 텐서
        # [2, 2*3, 2*3*4]    # 누적 곱

        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
        #누적 알파 값에 패딩을 추가하여 현재 누적 알파 값의 이전 값을 생성
        #이 값은 posterior 분포에서 사용
        # (1, 0)은 앞쪽에 1개를 패딩하고, 뒷쪽에는 0개를 패딩하라는 의미
        # [2, 3, 4]
        # 출력: tensor([1, 2, 3])
        #https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbgtvcW%2FbtsCRflvGlU%2FwQB54ftoZyKZKxJJ3Kk0Kk%2Fimg.png
        timesteps, = betas.shape
        #betas의 shape로부터 시간 단계의 수를 가져옴
        self.num_timesteps = int(timesteps)
        #시간 단계의 수를 정수형으로 변환하여 self.num_timesteps에 저장

        # 샘플링 관련 파라미터 설정
        self.sampling_timesteps = default(sampling_timesteps, timesteps)  # 샘플링 타임스텝 수 설정 # 주어진 값이 있으면 사용하고, 없으면 전체 타임스텝 수로 기본값 설정
        assert self.sampling_timesteps <= timesteps # 샘플링 타임스텝 수가 전체 타임스텝 수보다 작거나 같은지 확인
        self.is_ddim_sampling = self.sampling_timesteps < timesteps # 샘플링 타임스텝이 전체 타임스텝보다 작으면 다중 타임스텝 샘플링 사용 여부 설정
        self.ddim_sampling_eta = ddim_sampling_eta # 다중 타임스텝 샘플링에 대한 eta 매개변수 설정


        # 버퍼 등록을 위한 도우미 함수
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        # Beta 및 관련 값 버퍼 등록
        ## register_buffer를 써야하는 이유: https://www.ai-bio.info/pytorch-register-buffer
        # 모델 파라미터가 아닌 일시적인 상태 정보를 유지하기 위해 버퍼 등록
        register_buffer('betas', betas)  # beta 값 버퍼 등록
        register_buffer('alphas_cumprod', alphas_cumprod)  # alpha의 누적 곱 버퍼 등록
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)  # 이전 alpha 누적 곱 버퍼 등록

        # 확산 관련 계산
        # 다양한 계산에 사용될 수 있는 미리 계산된 값들을 버퍼로 등록
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))  # alpha 누적 곱의 제곱근 버퍼 등록 # forward process의 q(Xt|X0)
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))  # 1에서 alpha 누적 곱을 뺀 값의 제곱근 버퍼 등록# forward process의 q(Xt|X0)
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))  # 1에서 alpha 누적 곱을 뺀 값의 로그 버퍼 등록
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))  # alpha 누적 곱의 역수의 제곱근 버퍼 등록
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))  # alpha 누적 곱의 역수에서 1을 뺀 값의 제곱근 버퍼 등록


        # posterior q(x_{t-1} | x_t, x_0)을 위한 계산
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod) #논문에서는 간단하게 beta_t를 분산으로 쓰는 것도 제안함 -> beta-t

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        register_buffer('posterior_variance', posterior_variance)

        ## 아래: 확산 체인의 시작 부분에서 후방 분산이 0이기 때문에 로그 계산이 잘림
        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))
        #https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FSE4AN%2FbtsCUZvhdrP%2Fx2FvymfYRWdZBrGCJQ1ID0%2Fimg.png
        # register_buffer를 통해서, alphas_cumprod를 정의하고, 그것을 바탕으로 평균과, 표준편차에 이용되는 값들을 정의하고, posterior_variance를 통해서 분산도 정의


        # Offset 노이즈 강도
        self.offset_noise_strength = offset_noise_strength

        # SNR 계산 및 손실 가중치 유도
        # snr - signal noise ratio
        #DDPM의 loss를 계산할 때, target값을 무엇으로 할지 정하는 코드
        #기존 논문에서는 eta(noise)값을 예측하는 방향으로 수식을 설계했는데 다른 관점도 있음
        #  x0의 값을 예측하는 방법, velocity V를 예측하는 방법
        # default 값은 velocity V를 예측하는 방법이었는데, 아래에서 V가 무엇인지 살펴면
        # 추가적으로 t값에 따라 loss_weight를 설정하는 부분이 있음
        #해당 부분은 signal noise ratio를 이용하여 설정하게 되는데, 아래에 어떻게 이용되는지 설명이 나와있음 (SNR관련 자료: https://ko.wikipedia.org/wiki/%EC%8B%A0%ED%98%B8_%EB%8C%80_%EC%9E%A1%EC%9D%8C%EB%B9%84#) 

        snr = alphas_cumprod / (1 - alphas_cumprod)
        maybe_clipped_snr = snr.clone()

        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max=min_snr_gamma)

        if objective == 'pred_noise':
            register_buffer('loss_weight', maybe_clipped_snr / snr)
        elif objective == 'pred_x0':
            register_buffer('loss_weight', maybe_clipped_snr)
        elif objective == 'pred_v':
            register_buffer('loss_weight', maybe_clipped_snr / (snr + 1))

        # 데이터 자동 정규화 (0, 1) -> (-1, 1)
        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity

    @property
    def device(self):
        return self.betas.device

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, x_self_cond=None, clip_x_start=False, rederive_pred_noise=False):
        model_output = self.model(x, t, x_self_cond)
        maybe_clip = partial(torch.clamp, min=-1., max=1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

            if clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, x_self_cond=None, clip_denoised=True):
        preds = self.model_predictions(x, t, x_self_cond)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.inference_mode()
    def p_sample(self, x, t: int, x_self_cond=None):
        b, *_, device = *x.shape, self.device
        batched_times = torch.full((b,), t, device=device, dtype=torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x=x, t=batched_times, x_self_cond=x_self_cond,
                                                                          clip_denoised=True)
        noise = torch.randn_like(x) if t > 0 else 0.  # t == 0이면 노이즈 없음
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.inference_mode()
    def p_sample_loop(self, shape, return_all_timesteps=False):
        batch, device = shape[0], self.device

        img = torch.randn(shape, device=device)
        imgs = [img]

        x_start = None

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, t, self_cond)
            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim=1)

        ret = self.unnormalize(ret)
        return ret

    @torch.inference_mode()
    def ddim_sample(self, shape, return_all_timesteps=False):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        img = torch.randn(shape, device=device)
        imgs = [img]

        x_start = None

        for time, time_next in tqdm(time_pairs, desc='sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, self_cond, clip_x_start=True,
                                                             rederive_pred_noise=True)

            if time_next < 0:
                img = x_start
                imgs.append(img)
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim=1)

        ret = self.unnormalize(ret)
        return ret

    @torch.inference_mode()
    def sample(self, batch_size=16, return_all_timesteps=False):
        image_size, channels = self.image_size, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn((batch_size, channels, image_size, image_size), return_all_timesteps=return_all_timesteps)

    @torch.inference_mode()
    def interpolate(self, x1, x2, t=None, lam=0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.full((b,), t, device=device)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2

        x_start = None

        for i in tqdm(reversed(range(0, t)), desc='interpolation sample time step', total=t):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, i, self_cond)

        return img

    @autocast(enabled=False)
    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start)) #  alphas_cumprod = torch.cumprod(alphas, dim=0) # t개만큼의 누적곱

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + #alphas_cumprod = torch.cumprod(alphas, dim=0) # t개만큼의 누적곱
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise ## torch.sqrt(1. - alphas_cumprod)# torch.sqrt(1. - alphas_cumprod)
        )## (평균*x0 + 표준편차*eta) # eta ~ N(0,1) ==> Diffusino 논문에서 소개하는 q(Xt | X0) 수식
        # https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbsyTln%2FbtsCP5KfQab%2FmsComgfRDO9bX6nqRTnKx0%2Fimg.png q-sample 
    def p_losses(self, x_start, t, noise=None, offset_noise_strength=None):
        b, c, h, w = x_start.shape

        noise = default(noise, lambda: torch.randn_like(x_start)) # # randn_like -> 정규분포 따름 (diffusion process이므로 당연!) # 이부분이 noise 만드는 부분 같음

        # offset noise # offset noise - https://www.crosslabs.org/blog/diffusion-with-offset-noise
        #노이즈를 조절하기 위해 도입된 개념
        offset_noise_strength = default(offset_noise_strength, self.offset_noise_strength)

        if offset_noise_strength > 0.:
            offset_noise = torch.randn(x_start.shape[:2], device=self.device)
            noise += offset_noise_strength * rearrange(offset_noise, 'b c -> b c 1 1') # 이 부분을 변형하면 될거

        # noise sample
        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        # self-conditioning 여부에 따라, 50% 확률로 현재 시간에서 x_start를 예측하고 그로부터 unet으로 조건화
        # 이 기술은 훈련 속도를 25% 늦추지만, FID를 상당히 낮추는 것으로 보임

        x_self_cond = None
        if self.self_condition and random() < 0.5: # self_condition 나중에 살펴보기 // unet에서 정의할 때는 False
            with torch.no_grad():
                x_self_cond = self.model_predictions(x, t).pred_x_start
                x_self_cond.detach_()

        # 예측 및 그라디언트 단계 수행
        # model에 넣기 -> unet + attention
        model_out = self.model(x, t, x_self_cond)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss = F.mse_loss(model_out, target, reduction='none')
        loss = reduce(loss, 'b ... -> b', 'mean')

        loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean()

    def forward(self, img, *args, **kwargs):
        b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        img = self.normalize(img)
        return self.p_losses(img, t, *args, **kwargs)
    #image의 batch size만큼 t를 추출
    #그 이후 t값을 이미지와 함께 reverse process에 넣음