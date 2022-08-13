import torch
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
    model.cuda()
    model.eval()
    return model


def get_model(config, ckpt):
    global model
    if model is None:        
        config_yaml = OmegaConf.load(f"{config}")
        model = load_model_from_config(config_yaml, f"{ckpt}")
        model = model.to(device)
    return model


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model = None
inpainting_model = None
config = None




