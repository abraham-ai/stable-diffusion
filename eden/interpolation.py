import numpy as np
import torch
from einops import rearrange
from pytorch_lightning import seed_everything

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

from models import *


def slerp(t, v0, v1, DOT_THRESHOLD=0.9995):
    v0 = v0.detach().cpu().numpy()
    v1 = v1.detach().cpu().numpy()
    dot = np.sum(v0 * v1 / (np.linalg.norm(v0) * np.linalg.norm(v1)))
    if np.abs(dot) > DOT_THRESHOLD:
        v2 = (1 - t) * v0 + t * v1
    else:
        theta_0 = np.arccos(dot)
        sin_theta_0 = np.sin(theta_0)
        theta_t = theta_0 * t
        sin_theta_t = np.sin(theta_t)
        s0 = np.sin(theta_0 - theta_t) / sin_theta_0
        s1 = sin_theta_t / sin_theta_0
        v2 = s0 * v0 + s1 * v1
    return torch.from_numpy(v2).to(device)


def interpolate(opt, callback=None):
    model = get_model(opt.config, opt.ckpt)

    if opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)
    
    batch_size = opt.n_samples

    prompts = opt.interpolation_texts
    assert prompts and len(prompts) > 1

    start_code = None
    if opt.fixed_code:
        start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)

    with torch.no_grad():
        with model.ema_scope():

            uc = None
            if opt.scale != 1.0:
                uc = model.get_learned_conditioning(batch_size * [""])
            
            c_array = []
            for prompt in prompts:
                if isinstance(prompt, tuple):
                    prompt = list(prompt)
                c_array.append(model.get_learned_conditioning(prompt))

            # complete loop. if 2 prompts, can do this by copying frames (later)
            if len(prompts) > 2:
                c_array.append(c_array[0])

            shape = [opt.C, opt.H//opt.f, opt.W//opt.f]
            fs = np.linspace(0, 1, opt.n_interpolate)

            seed_everything(opt.seed)

            all_samples = list()
            
            for c0, c1 in zip(c_array[:-1], c_array[1:]):
                for f in fs:
                    c = slerp(f, c0, c1)

                    samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                     conditioning=c,
                                                     batch_size=opt.n_samples,
                                                     shape=shape,
                                                     verbose=False,
                                                     unconditional_guidance_scale=opt.scale,
                                                     unconditional_conditioning=uc,
                                                     eta=opt.ddim_eta,
                                                     dynamic_threshold=opt.dyn,
                                                     x_T=start_code)

                    x_samples_ddim = model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)

                    for x_sample in x_samples_ddim:
                        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                        all_samples.append(x_sample.astype(np.uint8))
                        if callback:
                            callback(x_sample)

    # add boomerang by reversing if only 2 prompts
    if len(prompts) == 2:
        for sample in reversed(all_samples):
            all_samples.append(sample)

    return all_samples
    