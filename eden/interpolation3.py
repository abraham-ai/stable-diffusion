####################
### Experimental ###

import random
import numpy as np
import PIL
import torch
from einops import rearrange, repeat
from tqdm import tqdm, trange
import cv2

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

from models import *
from interpolator import Interpolator

# opt = loop

texts = [
    'A green eyed furby looking mean',
    'Steampunk robot face',
    'Noir detective with cyclops eye',
    'A baby moose riding a dolphin'
]

The best thing about this job is that your bugs 

texts = [
    "city street with tall buildings and flying cars in the sky, cyberpunk environment artist, polycount, panfuturism, outrun, #screenshotsaturday, retrowave",
    "a futuristic city at night with neon lights, cyberpunk art by Ilya Kuvshinov, Artstation, retrofuturism, synthwave, retrowave, cityscape",
    "a neon city street filled with lots of neon signs, cyberpunk art by Liam Wong, featured on unsplash, panfuturism, anime aesthetic, glowing neon, synthwave",
    "a futuristic city with neon lights and buildings, a 3D render by Victor Mosquera, featured on polycount, cubo-futurism, rendered in cinema4d, rendered in unreal engine, octane render",
    "a computer generated image of a futuristic landscape, a matte painting by Mike Winkelmann, cg society contest winner, retrofuturism, synthwave, retrowave, outrun",
    "a painting of a mountain with a lake below it, an art deco painting by Lawren Harris, trending on behance, crystal cubism, fauvism, cubism, chillwave",
    "a painting of a mountain with a lake in front of it, a cubist painting by RHADS, pexels contest winner, crystal cubism, matte drawing, oil on canvas, dystopian art",
    "a black and white drawing of a mountain range, an ambient occlusion render by Buckminster Fuller, behance contest winner, generative art, outlined art, flat shading, low poly",
    "a black and white drawing of mountains, lineart by Otto Eckmann, behance contest winner, generative art, outlined art, flat shading, stipple"
]


opt = StableDiffusionSettings(
    config = config,
    ckpt = ckpt,
    mode = "interpolate",
    text_input = 'Steampunk robot face',
    interpolation_texts = random.sample(texts, 1) * 4,
    n_interpolate = 50,
    strength = 0.5,
    ddim_steps = 100,
    plms = False,
    H = 512,
    W = 896,
    seed = 19,
    fixed_code=True)



def load_img(img):
    if isinstance(img, str):
        image = Image.open(img).convert("RGB")
    else:
        image = Image.fromarray(img).convert("RGB")
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.


model = get_model(opt.config, opt.ckpt)

sampler = DDIMSampler(model)

batch_size = opt.n_samples

prompts = opt.interpolation_texts
assert prompts and len(prompts) > 1

# complete loop. if 2 prompts, can do this by copying frames (later)
if len(prompts) > 2:
    prompts.append(prompts[-1])


start_code = None
if opt.fixed_code:
    start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)

shape = [opt.C, opt.H//opt.f, opt.W//opt.f]

seeds = [random.randint(1, 1e8) for i in range(len(prompts))]

interpolator = Interpolator(model, prompts, opt.n_interpolate, opt, device, seeds=seeds)
imgs, ts = [], []
prev_img = None


margin = 1
f_idx = 1

with torch.no_grad():
    with model.ema_scope(): 
        uc = None
        if opt.scale != 1.0:
            uc = model.get_learned_conditioning(batch_size * [""])        
        for t in interpolator.ts:
            t, t_raw, c, start_code, scale, current_prompt, current_seed = interpolator.get_next_conditioning(smooth=0)
            if prev_img is None:
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
            else:
                init_image = load_img(prev_img).to(device)
                init_image = repeat(init_image, '1 ... -> b ...', b=opt.n_samples)
                init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))  # move to latent space
                sampler.make_schedule(ddim_num_steps=opt.ddim_steps, ddim_eta=opt.ddim_eta, verbose=False)
                t_enc = int(opt.strength * opt.ddim_steps)
                z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc]*batch_size).to(device))
                samples = sampler.decode(
                    z_enc, 
                    c, 
                    t_enc, 
                    unconditional_guidance_scale=opt.scale,
                    unconditional_conditioning=uc
                )          
            x_samples = model.decode_first_stage(samples)
            x_samples = torch.clamp((x_samples+1.0)/2.0, min=0.0, max=1.0)
            x_sample = 255. * rearrange(x_samples[0].cpu().numpy(), 'c h w -> h w c')
            x_sample = x_sample.astype(np.uint8)
            interpolator.add_frame(x_samples, t)
            ts.append(t_raw)
            imgs.append(x_sample)
            Image.fromarray(x_sample).save('frames/f%05d.png'%f_idx)
            f_idx+=1
            prev_img = x_sample
            prev_img = cv2.resize(prev_img[margin:-margin, margin:-margin, :], (opt.W, opt.H))
            #if callback:
            #    callback(x_sample)

imgs = [img for _, img in sorted(zip(ts, imgs))] 

# add boomerang by reversing if only 2 prompts
if len(prompts) == 2:
    for img in reversed(imgs):
        imgs.append(img)

