import random
import cv2
from PIL import Image
import moviepy.editor as mpy

import sys
sys.path.append('..')
sys.path.append('../MiDaS/')

from settings import StableDiffusionSettings
from generation import *
from interpolation3 import *
from inpainting import *
from img2img import img2img


config = "../../configs/stable-diffusion/v1_improvedaesthetics.yaml"
ckpt = "../models/v1pp-flatlined-hr.ckpt"    


opt = StableDiffusionSettings(
    config = config,
    ckpt = ckpt,
    mode = "interpolate",
    text_input = 'Steampunk robot face',
    interpolation_texts = [
        'A green eyed furby looking mean',
        'Steampunk robot face',
        'Noir detective with cyclops eye',
        'A baby moose riding a dolphin'
    ],
    n_interpolate = 50,
    ddim_steps = 25,
    plms = False,
    H = 512,
    W = 512,
    seed = 13,
    fixed_code=True)

frames = interpolate(opt)
clip = mpy.ImageSequenceClip(frames, fps=15)
clip.write_videofile('example_interpolation7.mp4')

