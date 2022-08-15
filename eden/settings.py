from dataclasses import dataclass, field
from typing import List
from PIL import Image


@dataclass
class StableDiffusionSettings:

    mode: str = "generate"

    config: str = "../configs/stable-diffusion/v1_improvedaesthetics.yaml"
    ckpt: str = "models/v1pp-flatlined-hr.ckpt"    

    text_input: str = "hello world" 
    
    combined_text_inputs: List = field(default_factory=lambda: [])
    combined_text_ratios: List = field(default_factory=lambda: [])

    # only for interpolate mode
    interpolation_texts: List = field(default_factory=lambda: [
        "hello world",
        "a painting of a virus monster playing guitar"
    ])
    n_interpolate: int = 1
    strength: float = 0.5
    
    # only for inpainting mode
    input_image: Image = None
    mask_image: Image = None

    # generation params
    seed: int = 13
    fixed_code: bool = False
    ddim_steps: int = 50
    plms: bool = False
    ddim_eta: float = 0.0
    C: int = 4
    f: int = 8    
    scale: float = 12.5
    dyn: float = None

    # dimensions, quantity
    H: int = 256
    W: int = 256
    n_iter: int = 1
    n_samples: int = 1
