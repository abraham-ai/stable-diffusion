import numpy as np
import torch
from einops import rearrange
from tqdm import tqdm, trange
from pytorch_lightning import seed_everything

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

from models import *


def get_embedding(opt):
    model = get_model(opt.config, opt.ckpt)
    with torch.no_grad():
        with model.ema_scope():
            c = model.get_learned_conditioning([opt.text_input])
            return c


def generate(opt, callback=None, update_image_every=1):
    model = get_model(opt.config, opt.ckpt)
    
    if opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)
    
    batch_size = opt.n_samples

    assert opt.text_input is not None \
        or opt.combined_text_inputs
        
    start_code = None
    if opt.fixed_code:
        start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)

    seed_everything(opt.seed)

    all_samples = list()

    def inner_callback(img, i):
        intermediate_samples = None
        if i % update_image_every != 0:
            intermediate_samples = []
            x_samples_ddim = model.decode_first_stage(img)
            x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)
            for x_sample in x_samples_ddim:
                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                intermediate_samples.append(x_sample.astype(np.uint8))
        callback(intermediate_samples)
    
    with torch.no_grad():
        with model.ema_scope():
            for n in trange(opt.n_iter, desc="Sampling"):

                uc = None
                if opt.scale != 1.0:
                    uc = model.get_learned_conditioning(batch_size * [""])

                if opt.combined_text_inputs:
                    w = torch.tensor(opt.combined_text_ratios).to(device)
                    cs = model.get_learned_conditioning(opt.combined_text_inputs)
                    c = torch.movedim(cs, 0, 2).multiply(w).sum(axis=-1).unsqueeze(0)
                else:
                    c = model.get_learned_conditioning([opt.text_input])

                shape = [opt.C, opt.H//opt.f, opt.W//opt.f]

                samples_ddim, _ = sampler.sample(
                    S=opt.ddim_steps,
                    img_callback=inner_callback if callback else None,
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

    return all_samples
    
