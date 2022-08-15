
################################
#### This doesn't work yet #####


#git clone https://github.com/isl-org/MiDaS.git
#mv MiDaS/utils.py MiDaS/midas_utils.py
# pip install timm
# wget https://github.com/intel-isl/DPT/releases/download/1_0/dpt_large-midas-2f21e586.pt -O models/dpt_large-midas-2f21e586.pt


import sys
sys.path.append('./MiDaS/')

import os
import math
import cv2
from PIL import Image
import numpy as np
import torch
import torchvision
import torchvision.transforms as T
import py3d_tools as p3d
import midas_utils
from midas.dpt_depth import DPTDepthModel
from midas.midas_net import MidasNet
from midas.midas_net_custom import MidasNet_small
from midas.transforms import Resize, NormalizeImage, PrepareForNet

device = torch.device('cuda:0')# if (torch.cuda.is_available() and not useCPU) else 'cpu')

model_path = 'models'


@torch.no_grad()
def transform_image_3d(img_filepath, midas_model, midas_transform, device, rot_mat=torch.eye(3).unsqueeze(0), translate=(0.,0.,-0.04), near=2000, far=20000, fov_deg=60, padding_mode='border', sampling_mode='bicubic', midas_weight = 0.3,spherical=False):
    img_pil = Image.open(open(img_filepath, 'rb')).convert('RGB')
    w, h = img_pil.size
    image_tensor = torchvision.transforms.functional.to_tensor(img_pil).to(device)

    # use_adabins = midas_weight < 1.0
    # use_adabins = False
    # if use_adabins:
    #     # AdaBins
    #     """
    #     predictions using nyu dataset
    #     """
    #     print("Running AdaBins depth estimation implementation...")
    #     infer_helper = InferenceHelper(dataset='nyu', device=device)

    #     image_pil_area = w*h
    #     if image_pil_area > MAX_ADABINS_AREA:
    #         scale = math.sqrt(MAX_ADABINS_AREA) / math.sqrt(image_pil_area)
    #         depth_input = img_pil.resize((int(w*scale), int(h*scale)), Image.LANCZOS) # LANCZOS is supposed to be good for downsampling.
    #     elif image_pil_area < MIN_ADABINS_AREA:
    #         scale = math.sqrt(MIN_ADABINS_AREA) / math.sqrt(image_pil_area)
    #         depth_input = img_pil.resize((int(w*scale), int(h*scale)), Image.BICUBIC)
    #     else:
    #         depth_input = img_pil
    #     try:
    #         _, adabins_depth = infer_helper.predict_pil(depth_input)
    #         if image_pil_area != MAX_ADABINS_AREA:
    #             adabins_depth = torchvision.transforms.functional.resize(torch.from_numpy(adabins_depth), image_tensor.shape[-2:], interpolation=torchvision.transforms.functional.InterpolationMode.BICUBIC).squeeze().to(device)
    #         else:
    #             adabins_depth = torch.from_numpy(adabins_depth).squeeze().to(device)
    #         adabins_depth_np = adabins_depth.cpu().numpy()
    #     except:
    #         pass

    torch.cuda.empty_cache()

    # MiDaS
    img_midas = midas_utils.read_image(img_filepath)
    img_midas_input = midas_transform({"image": img_midas})["image"]
    midas_optimize = True

    # MiDaS depth estimation implementation
    print("Running MiDaS depth estimation implementation...")
    sample = torch.from_numpy(img_midas_input).float().to(device).unsqueeze(0)
    if midas_optimize==True and device == torch.device("cuda"):
        sample = sample.to(memory_format=torch.channels_last)  
        sample = sample.half()
    prediction_torch = midas_model.forward(sample)
    prediction_torch = torch.nn.functional.interpolate(
            prediction_torch.unsqueeze(1),
            size=img_midas.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    prediction_np = prediction_torch.clone().cpu().numpy()

    print("Finished depth estimation.")
    torch.cuda.empty_cache()

    # MiDaS makes the near values greater, and the far values lesser. Let's reverse that and try to align with AdaBins a bit better.
    prediction_np = np.subtract(50.0, prediction_np)
    prediction_np = prediction_np / 19.0

    # if use_adabins:
    #     adabins_weight = 1.0 - midas_weight
    #     depth_map = prediction_np*midas_weight + adabins_depth_np*adabins_weight
    # else:
    #    depth_map = prediction_np
    depth_map = prediction_np

    depth_map = np.expand_dims(depth_map, axis=0)
    depth_tensor = torch.from_numpy(depth_map).squeeze().to(device)

    pixel_aspect = 1.0 # really.. the aspect of an individual pixel! (so usually 1.0)
    persp_cam_old = p3d.FoVPerspectiveCameras(near, far, pixel_aspect, fov=fov_deg, degrees=True, device=device)
    persp_cam_new = p3d.FoVPerspectiveCameras(near, far, pixel_aspect, fov=fov_deg, degrees=True, R=rot_mat, T=torch.tensor([translate]), device=device)

    # range of [-1,1] is important to torch grid_sample's padding handling
    y,x = torch.meshgrid(torch.linspace(-1.,1.,h,dtype=torch.float32,device=device),torch.linspace(-1.,1.,w,dtype=torch.float32,device=device))
    z = torch.as_tensor(depth_tensor, dtype=torch.float32, device=device)
    xyz_old_world = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)

    # Transform the points using pytorch3d. With current functionality, this is overkill and prevents it from working on Windows.
    # If you want it to run on Windows (without pytorch3d), then the transforms (and/or perspective if that's separate) can be done pretty easily without it.
    xyz_old_cam_xy = persp_cam_old.get_full_projection_transform().transform_points(xyz_old_world)[:,0:2]
    xyz_new_cam_xy = persp_cam_new.get_full_projection_transform().transform_points(xyz_old_world)[:,0:2]

    offset_xy = xyz_new_cam_xy - xyz_old_cam_xy
    # affine_grid theta param expects a batch of 2D mats. Each is 2x3 to do rotation+translation.
    identity_2d_batch = torch.tensor([[1.,0.,0.],[0.,1.,0.]], device=device).unsqueeze(0)
    # coords_2d will have shape (N,H,W,2).. which is also what grid_sample needs.
    coords_2d = torch.nn.functional.affine_grid(identity_2d_batch, [1,1,h,w], align_corners=False)
    offset_coords_2d = coords_2d - torch.reshape(offset_xy, (h,w,2)).unsqueeze(0)

    if spherical:
        spherical_grid = get_spherical_projection(h, w, torch.tensor([0,0], device=device), -0.4,device=device)#align_corners=False
        stage_image = torch.nn.functional.grid_sample(image_tensor.add(1/512 - 0.0001).unsqueeze(0), offset_coords_2d, mode=sampling_mode, padding_mode=padding_mode, align_corners=True)
        new_image = torch.nn.functional.grid_sample(stage_image, spherical_grid,align_corners=True) #, mode=sampling_mode, padding_mode=padding_mode, align_corners=False)
    else:
        new_image = torch.nn.functional.grid_sample(image_tensor.add(1/512 - 0.0001).unsqueeze(0), offset_coords_2d, mode=sampling_mode, padding_mode=padding_mode, align_corners=False)

    img_pil = torchvision.transforms.ToPILImage()(new_image.squeeze().clamp(0,1.))

    torch.cuda.empty_cache()

    return img_pil

def get_spherical_projection(H, W, center, magnitude,device):  
    xx, yy = torch.linspace(-1, 1, W,dtype=torch.float32,device=device), torch.linspace(-1, 1, H,dtype=torch.float32,device=device)  
    gridy, gridx  = torch.meshgrid(yy, xx)
    grid = torch.stack([gridx, gridy], dim=-1)  
    d = center - grid
    d_sum = torch.sqrt((d**2).sum(axis=-1))
    grid += d * d_sum.unsqueeze(-1) * magnitude 
    return grid.unsqueeze(0)


# Initialize MiDaS depth model.
# It remains resident in VRAM and likely takes around 2GB VRAM.
# You could instead initialize it for each frame (and free it after each frame) to save VRAM.. but initializing it is slow.


def init_midas_depth_model(model_path, midas_model_type="dpt_large", optimize=True):
    midas_model = None
    net_w = None
    net_h = None
    resize_mode = None
    normalization = None

    default_models = {
        "midas_v21_small": f"{model_path}/midas_v21_small-70d6b9c8.pt",
        "midas_v21": f"{model_path}/midas_v21-f6b98070.pt",
        "dpt_large": f"{model_path}/dpt_large-midas-2f21e586.pt",
        "dpt_hybrid": f"{model_path}/dpt_hybrid-midas-501f0c75.pt",
        "dpt_hybrid_nyu": f"{model_path}/dpt_hybrid_nyu-2ce69ec7.pt",}


    print(f"Initializing MiDaS '{midas_model_type}' depth model...")
    # load network
    midas_model_path = default_models[midas_model_type]

    if midas_model_type == "dpt_large": # DPT-Large
        midas_model = DPTDepthModel(
            path=midas_model_path,
            backbone="vitl16_384",
            non_negative=True,
        )
        net_w, net_h = 384, 384
        resize_mode = "minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    elif midas_model_type == "dpt_hybrid": #DPT-Hybrid
        midas_model = DPTDepthModel(
            path=midas_model_path,
            backbone="vitb_rn50_384",
            non_negative=True,
        )
        net_w, net_h = 384, 384
        resize_mode="minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    elif midas_model_type == "dpt_hybrid_nyu": #DPT-Hybrid-NYU
        midas_model = DPTDepthModel(
            path=midas_model_path,
            backbone="vitb_rn50_384",
            non_negative=True,
        )
        net_w, net_h = 384, 384
        resize_mode="minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    elif midas_model_type == "midas_v21":
        midas_model = MidasNet(midas_model_path, non_negative=True)
        net_w, net_h = 384, 384
        resize_mode="upper_bound"
        normalization = NormalizeImage(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    elif midas_model_type == "midas_v21_small":
        midas_model = MidasNet_small(midas_model_path, features=64, backbone="efficientnet_lite3", exportable=True, non_negative=True, blocks={'expand': True})
        net_w, net_h = 256, 256
        resize_mode="upper_bound"
        normalization = NormalizeImage(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    else:
        print(f"midas_model_type '{midas_model_type}' not implemented")
        assert False

    midas_transform = T.Compose(
        [
            Resize(
                net_w,
                net_h,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method=resize_mode,
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            normalization,
            PrepareForNet(),
        ]
    )

    midas_model.eval()
    
    if optimize==True:
        if device == torch.device("cuda"):
            midas_model = midas_model.to(memory_format=torch.channels_last)  
            midas_model = midas_model.half()

    midas_model.to(device)

    print(f"MiDaS '{midas_model_type}' depth model initialized.")
    return midas_model, midas_transform, net_w, net_h, resize_mode, normalization








#######################







def do_3d_step(img_filepath, frame_num, midas_model, midas_transform):
  if True:
    translation_x = 1 #args.translation_x_series[frame_num]
    translation_y = 1 #args.translation_y_series[frame_num]
    translation_z = -1 #args.translation_z_series[frame_num]
    rotation_3d_x = 0.1 #args.rotation_3d_x_series[frame_num]
    rotation_3d_y = 0.2 #args.rotation_3d_y_series[frame_num]
    rotation_3d_z = 0.1 #args.rotation_3d_z_series[frame_num]
    print(
        f'translation_x: {translation_x}',
        f'translation_y: {translation_y}',
        f'translation_z: {translation_z}',
        f'rotation_3d_x: {rotation_3d_x}',
        f'rotation_3d_y: {rotation_3d_y}',
        f'rotation_3d_z: {rotation_3d_z}',
    )

  translate_xyz = [-translation_x*TRANSLATION_SCALE, translation_y*TRANSLATION_SCALE, -translation_z*TRANSLATION_SCALE]
  rotate_xyz_degrees = [rotation_3d_x, rotation_3d_y, rotation_3d_z]
  print('translation:',translate_xyz)
  print('rotation:',rotate_xyz_degrees)
  rotate_xyz = [math.radians(rotate_xyz_degrees[0]), math.radians(rotate_xyz_degrees[1]), math.radians(rotate_xyz_degrees[2])]
  rot_mat = p3d.euler_angles_to_matrix(torch.tensor(rotate_xyz, device=device), "XYZ").unsqueeze(0)
  print("rot_mat: " + str(rot_mat))
  next_step_pil = transform_image_3d(img_filepath, midas_model, midas_transform, device,
                                          rot_mat, translate_xyz, near_plane, far_plane,
                                          fov, padding_mode=padding_mode,
                                          sampling_mode=sampling_mode, midas_weight=midas_weight)
  return next_step_pil












midas_weight = 0.3
near_plane = 200
far_plane = 10000
fov = 40
padding_mode = 'border'
sampling_mode = 'bicubic'


TRANSLATION_SCALE = 1.0/200.0

midas_depth_model = "dpt_large"



# midas_weight = 0.3
# near_plane = 200
# far_plane = 10000
# fov = 40
# padding_mode = 'border'
# sampling_mode = 'bicubic'


# TRANSLATION_SCALE = 1.0/200.0

# midas_depth_model = "dpt_large"
# midas_model, midas_transform, midas_net_w, midas_net_h, midas_resize_mode, midas_normalization = init_midas_depth_model(midas_depth_model)



# im2 = do_3d_step('../scripts/dog.jpg', 1, midas_model, midas_transform)
# im2.save('dog2.jpg')




