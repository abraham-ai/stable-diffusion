import pytorch_lightning
import numpy as np
import torch
import time

from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
import lpips

device = "cuda" if torch.cuda.is_available() else "cpu"
lpips_perceptor = lpips.LPIPS(net='vgg').eval().to(device)     # good but slow
#lpips_perceptor = lpips.LPIPS(net='alex').eval().to(device)    # fast
#lpips_perceptor = lpips.LPIPS(net='squeeze').eval().to(device)  # fast


def slerp(t, v0, v1, DOT_THRESHOLD=0.9995):
    '''
    Spherical linear interpolation
    Args:
        t (float/np.ndarray): Float value between 0.0 and 1.0
        v0 (np.ndarray): Starting vector
        v1 (np.ndarray): Final vector
        DOT_THRESHOLD (float): Threshold for considering the two vectors as
                            colineal. Not recommended to alter this.
    Returns:
        v2 (np.ndarray): Interpolation vector between v0 and v1
    '''
    if not isinstance(v0, np.ndarray):
        inputs_are_torch = True
        input_device = v0.device
        v0 = v0.cpu().numpy()
        v1 = v1.cpu().numpy()

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

    if inputs_are_torch:
        v2 = torch.from_numpy(v2).to(input_device)

    return v2

class FrameBuffer():
    def __init__(self):
        self.clear_buffer(keep_last_frame = False)
        
    def __len__(self):
        return len(self.frames)
        
    def clear_buffer(self, keep_last_frame = True):
        if keep_last_frame and len(self) > 0:
            self.frames, self.ts = [self.frames[-1]], [0.0]
        else:
            self.frames, self.ts = [], []

    def add_frame(self, img, t):
        # Add the img in the right order:
        self.frames.append(img)
        self.ts.append(t)
        sort_indices = np.argsort(self.ts)

        self.ts     = [self.ts[i] for i in sort_indices]
        self.frames = [self.frames[i] for i in sort_indices]


class Interpolator():
    '''
    Utility class to interpolate between creations (prompts + seeds + scales)

    TODO allow for batching (currently only does single batch)!!

    Usage example:
    
    prompts = ['artwork of the number one', 'artwork of the number two']
    seeds   = [20, 42]
    n_inter = 24 
    interpolator = Interpolator(model, prompts, n_inter, opt, device, seeds = seeds)

    for t in interpolator.ts:
        c, init_noise_img, scale, current_prompt, current_seed = interpolator.get_next_conditioning()
        img = sampler.render_img(...)

    '''

    def __init__(self, model, prompts, n_frames_between_two_prompts, opt, device, seeds = None, scales = None, source_shape = None):
        self.n_frames_between_two_prompts = n_frames_between_two_prompts
        self.prompts, self.seeds, self.scales = prompts, seeds, scales
        self.n = len(self.prompts)
        self.interpolation_step, self.prompt_index = 0, 0
        self.n_frames = int(self.n_frames_between_two_prompts*(self.n-1)+self.n)
        self.ts = np.linspace(0, self.n - 1, self.n_frames)
        self.frame_buffer = FrameBuffer()
        self.clear_buffer_at_next_iteration = False
        self.prev_prompt_index = 0

        # Copy over noise vector from a different resolution:
        self.source_shape = source_shape

        if self.seeds is None:
            self.seeds = np.random.randint(0, high=9999, size=self.n)
        if self.scales is None:
            self.scales = [opt.scale] * self.n
            
        assert len(self.seeds) == len(self.prompts) 
        assert len(self.scales) == len(self.prompts)    

        # Interpolate the scale (in case it varies per prompt):
        raw_x = np.linspace(0, self.n-1, self.n)
        interpolated_x = np.linspace(0, self.n-1, self.n_frames)
        self.scales = np.interp(interpolated_x, raw_x, self.scales)

        # Get conditioning and noise vectors:
        self.prompt_conditionings, self.init_noises = [], []
        for i in range(len(prompts)):
            prompt_conditioning, init_noise = self.get_creation_conditions(model, self.prompts[i], opt, device, seed = self.seeds[i])
            self.prompt_conditionings.append(prompt_conditioning)
            self.init_noises.append(init_noise)

    def copy_over_noise_from_different_resolution(self, init_noise, opt, source_shape, seed = None):
        if seed is not None:
            pytorch_lightning.seed_everything(seed)
        init_noise_s = torch.randn([1, opt.C, source_shape[0] // opt.f, source_shape[1] // opt.f], device=device)

        h_diff, w_diff = opt.H // opt.f - source_shape[0] // opt.f, opt.W // opt.f - source_shape[1] // opt.f 
        
        if h_diff >= 0: # current frame is taller than source
            copy_h = source_shape[0] // opt.f
            start_h = h_diff // 2
            top_source, bottom_source = 0, copy_h
            top, bottom               = start_h, start_h + copy_h
        else: # current frame is less tall than source
            h_diff = -h_diff
            copy_h = opt.H // opt.f
            start_h = h_diff // 2
            top_source, bottom_source = start_h, start_h + copy_h
            top, bottom               = 0, copy_h

        if w_diff >= 0: # current frame is wider than source
            copy_w = source_shape[1] // opt.f
            start_w = w_diff // 2
            left_source, right_source = 0, copy_w
            left, right               = start_w, start_w + copy_w
        else: # current frame is less wide than source
            w_diff = -w_diff
            copy_w = opt.W // opt.f
            start_w = w_diff // 2
            left_source, right_source = start_w, start_w + copy_w
            left, right               = 0, copy_w

        print("Copying noise from HxW = %dx%d to %dx%d" %(source_shape[0], source_shape[1], opt.H, opt.W))
        print("top-bottom: source %d --> %d is copied to %d --> %d" %(top_source * opt.f, bottom_source* opt.f, top * opt.f, bottom * opt.f))
        print("left-right: source %d --> %d is copied to %d --> %d" %(left_source* opt.f, right_source* opt.f, left * opt.f, right * opt.f))

        init_noise[:,:, top:bottom, left:right] = init_noise_s[:,:, top_source:bottom_source, left_source:right_source]
        return init_noise

    def get_creation_conditions(self, model, prompt, opt, device, seed = None, reset_seed = True):
        '''
        given a prompt and it's seed, return the two conditioning vectors needed to
        exactly reproduce that img: prompt conditioning and init_noise
        '''
        prompt_conditioning = model.get_learned_conditioning([prompt])

        if seed is not None:
            pytorch_lightning.seed_everything(seed)
        init_noise = torch.randn([1, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)

        if self.source_shape is not None: # Copy over a fraction of the init_noise from a different resolution:
            init_noise = self.copy_over_noise_from_different_resolution(init_noise, opt, self.source_shape, seed)

        if reset_seed:
            pytorch_lightning.utilities.seed.reset_seed()

        return prompt_conditioning, init_noise
    
    def d(self, img1, img2):
        with torch.no_grad():
            ms_ssim_diff  = (1 - ms_ssim(img1, img2, data_range=2.0, size_average=True)).cpu().item()
            l2_pixel_diff = ((img1 - img2)**2).mean().cpu().item()
            lpips_diff = lpips_perceptor(img1, img2).mean().item()
            mssim_f = 0.33
            l2_f    = 0.5
            print("lpips: %.3f | ssim: %.3f | L2: %.3f" %(lpips_diff, mssim_f*ms_ssim_diff, l2_f*l2_pixel_diff))
            return mssim_f * ms_ssim_diff + l2_f * l2_pixel_diff + lpips_diff

    def get_distances(self):
        distances = []
        s = time.time()
        for i in range(len(self.frame_buffer)-1):
            distances.append(self.d(self.frame_buffer.frames[i],self.frame_buffer.frames[i+1]))

        print("Distance compute took %.2f s" %(time.time() - s))
        return distances

    def add_frame(self, img, t):
        self.frame_buffer.add_frame(img, t)

    def find_next_t(self):
        if len(self.frame_buffer) == 0: # Render first prompt
            return 0
        elif len(self.frame_buffer) == 1: # Render second prompt
            return 1.0
        elif len(self.frame_buffer) == 2: # Render midpoint
            return 0.5
        else: # Find the best location to render the next frame:
            distances = self.get_distances()
            max_d_index = np.argmax(distances)
            t_left, t_right = self.frame_buffer.ts[max_d_index], self.frame_buffer.ts[max_d_index+1]
            return (t_left + t_right) / 2

    def get_next_conditioning(self, smooth = 1, verbose = 0):
        '''
        This function should be called iteratively in a loop to yield
        consecutive conditioning signals for the diffusion model
        ''' 
        if smooth:
            
            if self.clear_buffer_at_next_iteration:
                self.frame_buffer.clear_buffer()
                self.clear_buffer_at_next_iteration = False

            self.prompt_index = int(self.ts[self.interpolation_step])
            t = self.find_next_t()
            
            if self.prompt_index > self.prev_prompt_index: # Last frame of this prompt
                self.clear_buffer_at_next_iteration = True
                self.prev_prompt_index = self.prompt_index
                self.prompt_index -= 1

            t_raw = t + self.prompt_index

        else:
            t_raw = self.ts[self.interpolation_step]
            self.prompt_index = int(t_raw)
            t = t_raw % 1

        # Get all conditioning signals:
        c = slerp(t, self.prompt_conditionings[self.prompt_index], self.prompt_conditionings[(self.prompt_index + 1) % self.n])           
        init_noise = slerp(t, self.init_noises[self.prompt_index], self.init_noises[(self.prompt_index + 1) % self.n])
        scale = self.scales[self.interpolation_step]

        return_index = self.prompt_index
        if smooth and t == 1.0:
            return_index += 1

        if verbose:
            print("Interpolation step: %d  (raw_ts_lookup: %.5f)" %(self.interpolation_step, self.ts[self.interpolation_step]))
            print("Interpolating %d --> %d at %.5f" %(self.prompt_index, self.prompt_index+1, t))
            print("Saving frame with prompt index: %d and t=%.5f" %(return_index, t_raw))

        self.interpolation_step += 1

        return t, t_raw, c, init_noise, scale, self.prompts[return_index], self.seeds[return_index]

