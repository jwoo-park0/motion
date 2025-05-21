from collections import defaultdict
import numpy as np
import einops
from utils.video_util import randn_tensor,add_noise
import torch 
from sklearn.decomposition import PCA

class QueryStore:
    def __init__(self, mode='store', t_range=[0, 1000], strength_start=1, strength_end=1):
        """
        Initialize an empty ActivationsStore
        """
        self.query_store = defaultdict(list)
        self.mode = mode
        self.t_range = t_range
        self.strengthes = np.linspace(strength_start, strength_end, (t_range[1] - t_range[0]) + 1)

    def set_mode(self, mode): # mode can be 'cache' or 'inject'
        self.mode = mode

    # def cache_query(self, query, place_in_unet: str):
    def cache_query(self, query):
        assert self.mode == 'cache' 
        self.query_store = query

    # def inject_query(self, query, place_in_unet, t):
    def inject_query(self, query):
        assert self.mode == 'inject'
        # if t >= self.t_range[0] and t <= self.t_range[1]:
            # relative_t = t - self.t_range[0]
            # strength = self.strengthes[relative_t]
            # new_query = strength * self.query_store + (1 - strength) * query
        new_query = self.strength * self.query_store + (1-self.strength) * query
        # else:
            # new_query = query
        return new_query
    
    def update_bytime(self, t):
        assert self.query_store is not None 
        if t >= self.t_range[0] and t <= self.t_range[1]:
            relative_t = t - self.t_range[0]
            strength = self.strengthes[relative_t]
            self.strength = strength

class QueryStore2:
    def __init__(self, mode='store', t_range=[0, 1000], strength_start=1, strength_end=1, alpha=1):
        """
        Initialize an empty ActivationsStore
        """
        self.query_store = defaultdict(list)
        self.mode = mode
        self.t_range = t_range
        self.alpha = alpha
        self.strengthes = np.linspace(strength_start, strength_end, (t_range[1] - t_range[0]) + 1)

    def set_mode(self, mode): # mode can be 'cache' or 'inject'
        self.mode = mode

    # def cache_query(self, query, place_in_unet: str):
    def cache_query(self, query, t):
        assert self.mode == 'cache' 
        fft = True
        if t >= self.t_range[0] and t <= self.t_range[1]:
            if fft:
                self.query_store[t] = fft_query(query,tau=7, alpha = self.alpha)
            else: 
                self.query_store[t] = query
                
    # def inject_query(self, query, place_in_unet, t):
    def inject_query(self, query, t):
        assert self.mode == 'inject'
        if t >= self.t_range[0] and t <= self.t_range[1]:
            relative_t = t - self.t_range[0]
            strength = self.strengthes[relative_t]
            # adain_queries = True
            # if adain_queries == True and t <= 761 : 
            #     self.query_store[t] = adain(self.query_store[t].to(query.device), query)
            residual = query - query.mean(dim=1,keepdim=True)
            if t >= 901:
                self.query_store[t] = residual + self.query_store[t].to(query.device)
            new_query = strength * self.query_store[t].to(query.device) + (1-strength) * query 
        else:
            new_query = query
        return new_query
    
    def update_bytime(self, t):
        assert self.query_store is not None 
        if t >= self.t_range[0] and t <= self.t_range[1]:
            relative_t = t - self.t_range[0]
            strength = self.strengthes[relative_t]
            self.strength = strength

def obtain_query(self, video_data, noise_step, generator=None):
    
    video_latents = self.encode_first_stage(video_data.to(self.device))
    # video_latents = self.vae.config.scaling_factor * video_latents
    video_latents = video_latents.unsqueeze(0)
    video_latents = einops.rearrange(video_latents, "b f c h w -> b c f h w")

    uncond_embeddings = self.get_learned_conditioning([""]) # batch_size * [""]
    generator = torch.Generator(device=self.device)
    generator.manual_seed(self.config.seed)
    step_t = int(noise_step)
    noise_sampled = randn_tensor(video_latents.shape, generator=generator, device=video_latents.device, dtype=video_latents.dtype)
    noisy_latents = add_noise(self, step_t, video_latents, noise_sampled)
        
    _ = self.model.diffusion_model(noisy_latents, step_t, context=uncond_embeddings)


def get_cnt(self, video_data, generator=None):
    
    video_latents = self.encode_first_stage(video_data.to(self.device))
    # video_latents = self.vae.config.scaling_factor * video_latents
    video_latents = video_latents.unsqueeze(0)
    video_latents = einops.rearrange(video_latents, "b f c h w -> b c f h w")

    # uncond_embeddings = self.get_learned_conditioning([""]) # batch_size * [""]
    # generator = torch.Generator(device=self.device)
    # generator.manual_seed(self.config.seed)
    # step_t = int(noise_step)
    # noise_sampled = randn_tensor(video_latents.shape, generator=generator, device=video_latents.device, dtype=video_latents.dtype)
    # noisy_latents = add_noise(self, step_t, video_latents, noise_sampled)
        
    # _ = self.model.diffusion_model(noisy_latents, step_t, context=uncond_embeddings)

    return video_latents.clone()
    
def querycache_unet_attention(unet,**kwargs):
    # replace the fwd function
    # count = 0
    for name, module in unet.named_modules():
        module_name = type(module).__name__
        # if 'CrossAttention' in module_name: 
        if module_name == 'CrossAttention' :    
        # if 'Temporal_CrossAttention' in module_name:
            module.set_querystore(QueryStore2(**kwargs))
            module.querystore.mode ='cache'
            # count+=1
            # print(module_name)
    # print(count)
    return unet

def queryinject_unet_attention(unet):
    # replace the fwd function
    for name, module in unet.named_modules():
        module_name = type(module).__name__
        # if 'CrossAttention' in module_name: 
        if module_name == 'CrossAttention' :
        # if 'Temporal_CrossAttention' in module_name:
            # module.set_querystore(QueryStore(*kwargs))
            module.querystore.mode ='inject'
            # print(module_name)
    return unet

def update_unet_query(unet,t):
    # replace the fwd function
    for name, module in unet.named_modules():
        module_name = type(module).__name__
        # if 'CrossAttention' in module_name: 
        if module_name == 'CrossAttention' :
        # if 'Temporal_CrossAttention' in module_name:
            # module.set_querystore(QueryStore(*kwargs))
            module.querystore.update_bytime(t)
            # print(module_name)
    return unet


def fft_query(feature, tau, alpha) : 
    
    fft_features = torch.fft.fft(feature, dim=0)
    num_frames = feature.shape[0]
    seq_index = num_frames // 2
    high_freq_indices = [idx for idx in range(num_frames) if seq_index - tau <=  idx  <=  seq_index + tau]
    low_freq_indices = [idx for idx in range(num_frames) if idx not in high_freq_indices]

    low_freq = fft_features[low_freq_indices, :, :]
    high_freq = fft_features[high_freq_indices, :, :]
    
    fft_features[low_freq_indices, :, :] =  low_freq 
    fft_features[high_freq_indices, :, :] = (alpha) * high_freq
    rescaled_query = torch.fft.ifft(fft_features, dim=0).real
    # sum_dim = rescaled_query.sum(dim=0, keepdim=True)
    # rescaled_query /= torch.clamp(sum_dim, min=1e-3)
    
    # rescaled_query = torch.clamp(rescaled_query, min=feature.min(), max=feature.max())
    rescaled_query = torch.clamp(rescaled_query, min=np.percentile(rescaled_query.numpy(),1), 
                                max=np.percentile(rescaled_query.numpy(),99))
    
    return rescaled_query

def adain(feat, ref_feat):
    
    def calc_mean_std(feat, eps: float=1e-5):
        feat_std = (feat.var(dim=-2,keepdims=True) + eps).sqrt()
        feat_mean = feat.mean(dim=-2, keepdims=True)
        return feat_mean, feat_std
    
    feat_mean, feat_std = calc_mean_std(feat)
    ref_mean, ref_std = calc_mean_std(ref_feat)
    
    return ref_std * ((feat - feat_mean) / feat_std) + ref_mean

# def destroy_spatial_structure(feat):
#     # feat: (frame, query_dim, feature_dim)
#     shuffled = feat.clone()
#     for t in range(feat.shape[0]):
#         idx = torch.randperm(feat.shape[1])
#         shuffled[t] = shuffled[t, idx]
#     return shuffled
