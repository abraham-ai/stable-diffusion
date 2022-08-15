import random
import numpy as np
import torch
from einops import rearrange

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

from models import *
from interpolator import Interpolator


def interpolate(opt, callback=None):
    model = get_model(opt.config, opt.ckpt)

    if opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    batch_size = opt.n_samples

    prompts = opt.interpolation_texts
    assert prompts and len(prompts) > 1

    if opt.loop and len(prompts) > 2:
        prompts.append(prompts[-1])

    start_code = None
    if opt.fixed_code:
        start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)

    shape = [opt.C, opt.H//opt.f, opt.W//opt.f]

    seeds = [random.randint(1, 1e8) for i in range(len(prompts))]

    interpolator = Interpolator(model, prompts, opt.n_interpolate, opt, device, seeds=seeds)
    imgs, ts = [], []

    with torch.no_grad():
        with model.ema_scope():
            
            uc = None
            if opt.scale != 1.0:
                uc = model.get_learned_conditioning(batch_size * [""])
            
            for t in interpolator.ts:
                t, t_raw, c, start_code, scale, current_prompt, current_seed = interpolator.get_next_conditioning(smooth=opt.smooth)
                
                samples, _ = sampler.sample(
                    S=opt.ddim_steps,
                    conditioning=c,
                    batch_size=opt.n_samples,
                    shape=shape,
                    verbose=False,
                    unconditional_guidance_scale=scale,
                    unconditional_conditioning=uc,
                    eta=opt.ddim_eta,
                    dynamic_threshold=opt.dyn,
                    x_T=start_code)
                
                x_samples = model.decode_first_stage(samples)
                x_samples = torch.clamp((x_samples+1.0)/2.0, min=0.0, max=1.0)
                x_sample = 255. * rearrange(x_samples[0].cpu().numpy(), 'c h w -> h w c')
                x_sample = x_sample.astype(np.uint8)

                interpolator.add_frame(x_samples, t)

                ts.append(t_raw)
                imgs.append(x_sample)
                
                if callback:
                    callback(x_sample)

    imgs = [img for _, img in sorted(zip(ts, imgs))] 
    
    # add boomerang by reversing if only 2 prompts
    if opt.loop and len(prompts) == 2:
        for img in reversed(imgs):
            imgs.append(img)

    return imgs
    
