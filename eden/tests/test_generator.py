import cv2
from PIL import Image
import moviepy.editor as mpy

import sys
sys.path.append('..')
from settings import StableDiffusionSettings
from generation import *
from interpolation import *
from inpainting import *

config = "../../configs/stable-diffusion/v1_improvedaesthetics.yaml"
ckpt = "../models/v1pp-flatlined-hr.ckpt"    


def test_generate():
    opt = StableDiffusionSettings(
        config = config,
        ckpt = ckpt,
        mode = "generate",
        text_input = 'A cyberpunk city',
        ddim_steps = 50,
        plms = True,
        H = 512,
        W = 512)

    results = generate(opt)
    Image.fromarray(results[0]).save('example.png')


def test_interpolate():
    opt = StableDiffusionSettings(
        config = config,
        ckpt = ckpt,
        mode = "interpolate",
        text_input = 'Steampunk robot face',
        interpolation_texts = [
            'Steampunk robot face',
            'Noir detective with cyclops eye'
        ],
        n_interpolate = 5,
        ddim_steps = 50,
        plms = True,
        H = 512,
        W = 512,
        seed = 13,
        fixed_code=True)

    frames = interpolate(opt)
    clip = mpy.ImageSequenceClip(frames, fps=20)
    clip.write_videofile('example_interpolation.mp4')


def test_inpainting():
    pass


test_generate()
test_interpolate()
test_inpainting()
