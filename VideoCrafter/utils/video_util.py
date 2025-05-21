import decord
import torch.nn.functional as F
import numpy as np
from einops import rearrange
import einops
import torch
from typing import List, Optional, Tuple, Union, Callable
from lvdm.models.utils_diffusion import timestep_embedding
# from diffusers.utils import deprecate, logging, BaseOutput
import logging
logger = logging.getLogger(__name__)
# logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

def randn_tensor(
    shape: Union[Tuple, List],
    generator: Optional[Union[List["torch.Generator"], "torch.Generator"]] = None,
    device: Optional["torch.device"] = None,
    dtype: Optional["torch.dtype"] = None,
    layout: Optional["torch.layout"] = None,
):
    """A helper function to create random tensors on the desired `device` with the desired `dtype`. When
    passing a list of generators, you can seed each batch size individually. If CPU generators are passed, the tensor
    is always created on the CPU.
    """
    # device on which tensor is created defaults to device
    rand_device = device
    batch_size = shape[0]

    layout = layout or torch.strided
    device = device or torch.device("cpu")

    if generator is not None:
        gen_device_type = generator.device.type if not isinstance(generator, list) else generator[0].device.type
        if gen_device_type != device.type and gen_device_type == "cpu":
            rand_device = "cpu"
            if device != "mps":
                logger.info(
                    f"The passed generator was created on 'cpu' even though a tensor on {device} was expected."
                    f" Tensors will be created on 'cpu' and then moved to {device}. Note that one can probably"
                    f" slighly speed up this function by passing a generator that was created on the {device} device."
                )
        elif gen_device_type != device.type and gen_device_type == "cuda":
            raise ValueError(f"Cannot generate a {device} tensor from a generator of type {gen_device_type}.")

    if isinstance(generator, list):
        shape = (1,) + shape[1:]
        latents = [
            torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype, layout=layout)
            for i in range(batch_size)
        ]
        latents = torch.cat(latents, dim=0).to(device)
    else:
        latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype, layout=layout).to(device)

    return latents

def video_preprocess(video_path, height, width, video_length, duration=None, sample_start_idx=0,):
    
    video_name = video_path.split('/')[-1].split('.')[0]
    vr = decord.VideoReader(video_path)
    fps = vr.get_avg_fps()
    if  duration is None:
        total_frames = len(vr)
    else:
        
        total_frames = int(fps * duration)
        total_frames = min(total_frames, len(vr))  
        
    sample_index = np.linspace(0, total_frames - 1, video_length, dtype=int)
    print(total_frames,sample_index)
    video = vr.get_batch(sample_index)
    video = rearrange(video.asnumpy(), "f h w c -> f c h w")
    video = torch.from_numpy(video)
    video = F.interpolate(video, size=(height, width), mode="bilinear", align_corners=True)
    
    # video_sample = rearrange(video, "(b f) c h w -> b f h w c", f=video_length)
    # imageio.mimwrite(f"processed_videos/sample_{video_name}.mp4", video_sample[0], fps=8, quality=9)
    
    video = video / 127.5 - 1.0
    
    return video


def add_noise(self, timestep, x_0, noise_pred):
    # alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
    alpha_prod_t = self.alphas_cumprod[timestep] 
    # alpha_prod_t = self.ddim_alphas[timestep] 
    beta_prod_t = 1 - alpha_prod_t
    latents_noise = alpha_prod_t ** 0.5 * x_0 + beta_prod_t ** 0.5 * noise_pred
    return latents_noise
    
@torch.no_grad()
def obtain_motion_representation(self, generator=None, motion_representation_path: str = None,
                                 duration=None,use_controlnet=False,):
    
    video_data = video_preprocess(self.config.video_path, self.config.height, 
                                  self.config.width, self.config.video_length, duration=duration)
    # video_latents = self.vae.encode(video_data.to(self.vae.dtype).to(self.vae.device)).latent_dist.mode()
    video_latents = self.encode_first_stage(video_data.to(self.device))
    # video_latents = self.vae.config.scaling_factor * video_latents
    video_latents = video_latents.unsqueeze(0)
    video_latents = einops.rearrange(video_latents, "b f c h w -> b c f h w")
    
    # uncond_input = self.tokenizer(
    #     [""], padding="max_length", max_length=self.tokenizer.model_max_length,
    #     return_tensors="pt"
    # )
    uncond_embeddings = self.get_learned_conditioning([""])
    step_t = int(self.config.add_noise_step)
    # uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]
    noise_sampled = randn_tensor(video_latents.shape, generator=generator, device=video_latents.device, dtype=video_latents.dtype)
    noisy_latents = add_noise(self, step_t, video_latents, noise_sampled)
    down_block_additional_residuals = mid_block_additional_residual = None
    
    if use_controlnet:
        controlnet_image_index = self.input_config.image_index 
        if self.controlnet.use_simplified_condition_embedding:
            controlnet_images = video_latents[:,:,controlnet_image_index,:,:] 
        else:
            controlnet_images = (einops.rearrange(video_data.unsqueeze(0).to(self.vae.dtype).to(self.vae.device), "b f c h w -> b c f h w")+1)/2
            controlnet_images = controlnet_images[:,:,controlnet_image_index,:,:]

        controlnet_cond_shape    = list(controlnet_images.shape)
        controlnet_cond_shape[2] = noisy_latents.shape[2]
        controlnet_cond = torch.zeros(controlnet_cond_shape).to(noisy_latents.device).to(noisy_latents.dtype)

        controlnet_conditioning_mask_shape    = list(controlnet_cond.shape)
        controlnet_conditioning_mask_shape[1] = 1
        controlnet_conditioning_mask          = torch.zeros(controlnet_conditioning_mask_shape).to(noisy_latents.device).to(noisy_latents.dtype)

        controlnet_cond[:,:,controlnet_image_index] = controlnet_images
        controlnet_conditioning_mask[:,:,controlnet_image_index] = 1

        down_block_additional_residuals, mid_block_additional_residual = self.controlnet(
            noisy_latents, step_t,
            encoder_hidden_states=uncond_embeddings,
            controlnet_cond=controlnet_cond,
            conditioning_mask=controlnet_conditioning_mask,
            conditioning_scale=self.input_config.controlnet_scale,
            guess_mode=False, return_dict=False,
        )

    # _ = self.unet(noisy_latents, step_t, encoder_hidden_states=uncond_embeddings, return_dict=False, only_motion_feature=True,
    #               down_block_additional_residuals = down_block_additional_residuals,
    #               mid_block_additional_residual = mid_block_additional_residual,)
    _ = self.model.diffusion_model(noisy_latents, step_t, context=uncond_embeddings)
                #   down_block_additional_residuals = down_block_additional_residuals,
                #   mid_block_additional_residual = mid_block_additional_residual,)
    temp_attn_prob_control = self.get_temp_attn_prob()   
    motion_representation = {key: [max_value, max_index.to(torch.uint8)] for key, tensor in temp_attn_prob_control.items() for max_value, max_index in [torch.topk(tensor, k=1, dim=-1)]} 
    
    torch.save(motion_representation, motion_representation_path)
    self.motion_representation_path = motion_representation_path


def get_temp_attn_prob(self, index_select=None):
        
    attn_prob_dic = {}

    for name, module in self.model.diffusion_model.named_modules():
        module_name = type(module).__name__
        if "Temporal_CrossAttention" in module_name and classify_blocks(self.config.motion_guidance_blocks, name):
            key = module.processor.key
            if index_select is not None:
                get_index = torch.repeat_interleave(torch.tensor(index_select), repeats=key.shape[0]//len(index_select))
                index_all = torch.arange(key.shape[0])
                index_picked = index_all[get_index.bool()]
                key = key[index_picked]
            key = module.reshape_heads_to_batch_dim(key).contiguous()
            
            query = module.processor.query
            if index_select is not None:
                query = query[index_picked]
            query = module.reshape_heads_to_batch_dim(query).contiguous()
            attention_probs = module.get_attention_scores(query, key, None)         
            attention_probs = attention_probs.reshape(-1, module.heads,attention_probs.shape[1], attention_probs.shape[2])            
            attn_prob_dic[name] = attention_probs

    return attn_prob_dic


class MySelfAttnProcessor:
    def __init__(self, attention_op: Optional[Callable] = None):
        self.attention_op = attention_op
        

    def __call__(self, attn, hidden_states, query, key, value, attention_mask):
        # self.attn = attn
        self.key = key
        self.query = query
        # self.value = value
        # self.attention_mask = attention_mask
        # self.hidden_state = hidden_states.detach()
        # return hidden_states
    
    def record_qkv(self, attn, hidden_states, query, key, value, attention_mask):
        # self.attn = attn
        self.key = key
        self.query = query
        # self.value = value
        # # self.attention_mask = attention_mask
        # self.hidden_state = hidden_states.detach()
        # # import pdb; pdb.set_trace()

    # def replace_q(self):
    #     if hasattr(self, 'query'):
    #         return self.query
    
    def record_attn_mask(self, attn, hidden_states, query, key, value, attention_mask):
        self.attn = attn
        self.attention_mask = attention_mask
        
def classify_blocks(block_list, name):
    is_correct_block = False
    for block in block_list:
        if block in name:
            is_correct_block = True
            break
    return is_correct_block


def prep_unet_attention(unet,motion_gudiance_blocks):
    # replace the fwd function
    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if "Temporal_CrossAttention" in module_name and classify_blocks(motion_gudiance_blocks, name): # the temporary attention in guidance blocks
            module.set_processor(MySelfAttnProcessor())
            print(module_name)
    return unet


def unet_customized_forward(self, x, timesteps, context=None, 
                            features_adapter=None, fps=16, **kwargs):
    if not torch.is_tensor(timesteps):
        # This would be a good case for the `match` statement (Python 3.10+)
        is_mps = x.device.type == "mps"
        if isinstance(timesteps, float):
            dtype = torch.float32 if is_mps else torch.float64
        else:
            dtype = torch.int32 if is_mps else torch.int64
        timesteps = torch.tensor([timesteps], dtype=dtype, device=x.device)
    elif len(timesteps.shape) == 0:
        timesteps = timesteps[None].to(x.device)

    # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
    timesteps = timesteps.expand(x.shape[0])
        
    t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
    emb = self.time_embed(t_emb)

    if self.fps_cond:
        if type(fps) == int:
            fps = torch.full_like(timesteps, fps)
        fps_emb = timestep_embedding(fps,self.model_channels, repeat_only=False)
        emb += self.fps_embedding(fps_emb)
    
    b,_,t,_,_ = x.shape
    ## repeat t times for context [(b t) 77 768] & time embedding
    context = context.repeat_interleave(repeats=t, dim=0)
    emb = emb.repeat_interleave(repeats=t, dim=0)

    ## always in shape (b t) c h w, except for temporal layer
    x = rearrange(x, 'b c t h w -> (b t) c h w')

    h = x.type(self.dtype)
    adapter_idx = 0
    hs = []
    
    # input block 
    for id, module in enumerate(self.input_blocks):
        h = module(h, emb, context=context, batch_size=b)
        if id ==0 and self.addition_attention:
            h = self.init_attn(h, emb, context=context, batch_size=b)
        ## plug-in adapter features
        if ((id+1)%3 == 0) and features_adapter is not None:
            h = h + features_adapter[adapter_idx]
            adapter_idx += 1
        hs.append(h)
    if features_adapter is not None:
        assert len(features_adapter)==adapter_idx, 'Wrong features_adapter'

    h = self.middle_block(h, emb, context=context, batch_size=b)
    
    
    #output_block 
    for i , module in enumerate(self.output_blocks):
        #(i) : gradient 
        if i <= int(self.config.motion_guidance_blocks[-1].split(".")[-1]):
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context=context, batch_size=b)
        #(ii) : no gradient 
        else:
            with torch.no_grad():
                h = torch.cat([h, hs.pop()], dim=1)
                h = module(h, emb, context=context, batch_size=b)
    h = h.type(x.dtype)
    y = self.out(h)
    
    # reshape back to (b c t h w)
    y = rearrange(y, '(b t) c h w -> b c t h w', b=b)
    return y




# def schedule_set_timesteps(self, num_inference_steps: int, guidance_steps: int = 0, guiduance_scale: float = 0.0, device: Union[str, torch.device] = None,timestep_spacing_type= "uneven"):
#     """
#     Sets the discrete timesteps used for the diffusion chain (to be run before inference).

#     Args:
#         num_inference_steps (`int`):
#             The number of diffusion steps used when generating samples with a pre-trained model.
#     """

#     if num_inference_steps > self.config.num_train_timesteps:
#         raise ValueError(
#             f"`num_inference_steps`: {num_inference_steps} cannot be larger than `self.config.train_timesteps`:"
#             f" {self.config.num_train_timesteps} as the unet model trained with this scheduler can only handle"
#             f" maximal {self.config.num_train_timesteps} timesteps."
#         )

#     self.num_inference_steps = num_inference_steps
    
#     # assign more steps in early denoising stage for motion guidance
#     if timestep_spacing_type == "uneven":
#         timesteps_guidance = (
#             np.linspace(int((1-guiduance_scale)*self.config.num_train_timesteps), self.config.num_train_timesteps - 1, guidance_steps)
#             .round()[::-1]
#             .copy()
#             .astype(np.int64)
#         )
#         timesteps_vanilla = (
#             np.linspace(0, int((1-guiduance_scale)*self.config.num_train_timesteps) - 1, num_inference_steps-guidance_steps)
#             .round()[::-1]
#             .copy()
#             .astype(np.int64)
#         )
#         timesteps = np.concatenate((timesteps_guidance, timesteps_vanilla))

#     # "linspace", "leading", "trailing" corresponds to annotation of Table 2. of https://arxiv.org/abs/2305.08891
#     elif timestep_spacing_type == "linspace":
#         timesteps = (
#             np.linspace(0, self.config.num_train_timesteps - 1, num_inference_steps)
#             .round()[::-1]
#             .copy()
#             .astype(np.int64)
#         )
#     elif timestep_spacing_type == "leading":
#         step_ratio = self.config.num_train_timesteps // self.num_inference_steps
#         # creates integer timesteps by multiplying by ratio
#         # casting to int to avoid issues when num_inference_step is power of 3
#         timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
#         timesteps += self.config.steps_offset
#     elif timestep_spacing_type == "trailing":
#         step_ratio = self.config.num_train_timesteps / self.num_inference_steps
#         # creates integer timesteps by multiplying by ratio
#         # casting to int to avoid issues when num_inference_step is power of 3
#         timesteps = np.round(np.arange(self.config.num_train_timesteps, 0, -step_ratio)).astype(np.int64)
#         timesteps -= 1
#     else:
#         raise ValueError(
#             f"{timestep_spacing_type} is not supported. Please make sure to choose one of 'leading' or 'trailing'."
#         )

#     self.timesteps = torch.from_numpy(timesteps).to(device)
