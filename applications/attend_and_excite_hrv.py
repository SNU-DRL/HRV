import inspect
from typing import Any, Callable, Dict, List, Optional, Union, Tuple

import math
import numbers
import numpy as np
import torch
import torch.nn.functional as nnf

from packaging import version
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import logging
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker

from diffusers import StableDiffusionPipeline
from torch import nn
import abc
import pprint
from PIL import Image
import os
import argparse
import csv
import warnings
from torchvision.transforms.functional import gaussian_blur
warnings.filterwarnings("ignore", category=UserWarning)

# Arguments
def get_args_parser():
    parser = argparse.ArgumentParser("Running Attend-and-Excite", add_help=False)

    # Dataset parameters
    parser.add_argument("--gpu_num", default=0, type=int)
    parser.add_argument('--prompt', type=str, default=None)
    parser.add_argument('--model_card', type=str, default="CompVis/stable-diffusion-v1-4") 
    parser.add_argument('--negative_prompt', action='store_true')
    parser.add_argument('--run_standard_attend_excite', action='store_true')
    parser.add_argument('--token_indices', nargs='+', type=int, default=None)
    parser.add_argument('--seeds', nargs='+', type=int, default=[42])
    parser.add_argument('--concepts', nargs='+', type=str, default=["Animals", "Animals"]) 
    parser.add_argument('--output_path', type=str, default='./applications/multi_concept_outputs')
    parser.add_argument('--n_inference_steps', type=int, default=50)
    parser.add_argument('--guidance_scale', type=float, default=7.5)
    parser.add_argument('--max_iter_to_alter', type=int, default=25)
    parser.add_argument('--attention_res', type=int, default=8) 
    parser.add_argument('--run_standard_sd', action='store_true')
    parser.add_argument('--thresholds', type=Dict[int, float], default={0: 0.05, 10: 0.5, 20: 0.8})
    parser.add_argument('--scale_factor', type=int, default=20)
    parser.add_argument('--scale_range', type=Tuple[float, float], default=(1., 0.5))
    parser.add_argument('--smooth_attentions', action='store_true')
    parser.add_argument('--sigma', type=float, default=0.5) 
    parser.add_argument('--kernel_size', type=int, default=3) 
    parser.add_argument('--save_cross_attention_maps', action='store_true')
    parser.add_argument('--experiment_num', type=int, default=None)
    parser.add_argument('--description_file_path', type=str, default="./descriptions/340_final_text_list.csv")
    parser.add_argument('--category_vectors_path', type=str, default="./final_result/category_vectors_epoch_1_to_1_neg_prompt_False_merged_340_final_text_descriptions_sd_v1_4.npy")
    parser.add_argument('--head_index_path', type=str, default="./final_result/head_index_epoch_1_to_1_neg_prompt_False_merged_340_final_text_descriptions_sd_v1_4.npy")
    return parser

args = get_args_parser()
args = args.parse_args()
if args.model_card == "CompVis/stable-diffusion-v1-4":
    args.sd_1_4 = True
    args.model_version = "sd_v1_4"
else:
    raise Exception("Not implemented yet")

if args.run_standard_attend_excite:
    # hyperparameters for standard Attend-and-Excite
    args.attention_res = 16
    args.sigma = 0.5
    args.kernel_size = 3
os.makedirs(args.output_path, exist_ok=True)

# Access information
LOW_RESOURCE = False
NUM_DIFFUSION_STEPS = args.n_inference_steps
MODEL_CARD = args.model_card 
device = torch.device("cuda:{}".format(args.gpu_num)) if torch.cuda.is_available() else torch.device("cpu")
args.concept_vectors = np.load(args.category_vectors_path)
args.head_index = np.load(args.head_index_path)

logger = logging.get_logger(__name__)

# --- define CONCEPTS --- #
data = []
with open(args.description_file_path, "r") as f:
    render = csv.reader(f)
    for row in render:
        data.append(row)
num_list = np.arange(1, 11).astype(str)
data = [row for row in data if row != []]
data = [row[0] for row in data if not any(num in row[0] for num in num_list)]
data[0] = data[0].replace("\ufeff","")

CONCEPTS = data
# --------------------------- #

def latent2image(vae, latents):
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents)['sample']
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).astype(np.uint8)
    return image

"""Define Gaussian Smoothing"""
class GaussianSmoothing(nn.Module):
    """ This class is modified from the following source: [Source] https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/9
    
    Apply gaussian smoothing on a
    1d or 2d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 1 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=1):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32).to(device)
                for size in kernel_size
            ], indexing='ij'
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = nnf.conv1d
        elif dim == 2:
            self.conv = nnf.conv2d
        else:
            raise RuntimeError(
                'Only 1 and 2 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups)

class AttentionControl(abc.ABC):
    
    def step_callback(self, x_t):
        return x_t
    
    def between_steps(self):
        return
    
    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if LOW_RESOURCE else 0
    
    @abc.abstractmethod
    def forward (self, attn, is_cross: bool, place_in_unet: str, layer, num_heads):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str, layer, num_heads):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            self.forward(attn, is_cross, place_in_unet, layer, num_heads)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn
    
    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

class EmptyControl(AttentionControl):
    
    def forward (self, attn, is_cross: bool, place_in_unet: str, layer):
        return attn
    
    
class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str, layer, num_heads):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 32 ** 2:  # avoid memory overhead
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        self.attention_store = self.step_store
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = self.attention_store
        return average_attention


    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

class AttentionStoreHead(AttentionControl):
    @staticmethod
    def get_empty_store(concepts):
        base_dict = {"down_cross": [], "mid_cross": [], "up_cross": [],
                      "down_self": [],  "mid_self": [],  "up_self": []}
        empty_dict = {concept: base_dict for concept in concepts}
        return empty_dict

    def forward(self, attn, is_cross: bool, place_in_unet: str, layer, num_heads):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        # Save attention maps only if head is in self.head_list
        if is_cross:
            if attn.shape[0] == 2*num_heads:
                return attn
            
            batch_size = attn.shape[0] // num_heads # attn.shape: (batch_size*h, res^2, 77)
            reshaped_attn = attn.reshape(batch_size, num_heads, *attn.shape[1:]) # (1, h, res^2, 77)
            res = int(math.sqrt(reshaped_attn.shape[2]))
            cross_maps = reshaped_attn.reshape(reshaped_attn.shape[1], res, res, reshaped_attn.shape[-1]) # (h, res, res, 77)
            if res == args.attention_res:
                pass
            elif res < args.attention_res:
                cross_maps = cross_maps.permute(0, 3, 1, 2) # (h, 77, res, res)
                cross_maps = nnf.interpolate(cross_maps, size=(args.attention_res, args.attention_res),
                                                       mode='bicubic', align_corners=False)
                cross_maps = cross_maps.permute(0, 2, 3, 1) # (h, res, res, 77)
            else:
                cross_maps = cross_maps.permute(0, 3, 1, 2) # (h, 77, res, res)
                # --- preventing space aliasing during downsampling --- #
                kernel_size_ = (res//args.attention_res)+1
                sigma_ = 0.3 * ((kernel_size_ - 1) * 0.5 - 1) + 0.8
                cross_maps = gaussian_blur(cross_maps, kernel_size=(kernel_size_, kernel_size_), sigma=(sigma_, sigma_))
                # ----------------------------------------------------- #
                cross_maps = nnf.interpolate(cross_maps, size=(args.attention_res, args.attention_res), mode='bicubic', align_corners=False)
                cross_maps = cross_maps.permute(0, 2, 3, 1) # (h, res, res, 77)

            if place_in_unet == "down" and layer == 0:
                self.averaged_maps = [0 for _ in range(len(args.concepts))]

            for idx, concept in enumerate(args.concepts):
                for head in range(num_heads):
                    head_pos = f"{place_in_unet}, layer: {layer}, head: {head}"
                    soft_rescale_factor = args.concept_vectors[CONCEPTS.index(concept), np.where(args.head_index == head_pos)[0][0]]
                    if args.model_version == "sd_v1_4":
                        self.averaged_maps[idx] += (soft_rescale_factor * cross_maps[head, :, :, :] / 128)
                    else:
                        raise Exception("Not implemented yet")
            return attn
        return attn
    
    def between_steps(self):
        pass
    
    def get_average_attention(self):
        average_attention = self.attention_store
        return average_attention
    
    def reset(self):
        super(AttentionStoreHead, self).reset()
        self.step_store = self.get_empty_store(self.concepts)
        self.attention_store = {}
    
    def __init__(self, concepts=None):
        super(AttentionStoreHead, self).__init__()
        self.concepts = concepts
        self.averaged_maps = [0 for _ in range(len(args.concepts))]


def aggregate_attention(attention_store: AttentionStore, res: int, from_where: List[str], is_cross: bool, select: int, prompts: List[str] = [""]):
    """For selected locations (from_where: 'down', 'mid', 'up') and selected resolutions (res: ex. 16)
       returns the average attention map for the selected prompt (sel: ex. 0)."""
    out = []
    attention_maps = attention_store.get_average_attention()
    num_pixels = res ** 2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(len(prompts), -1, res, res, item.shape[-1])[select]
                out.append(cross_maps)
    out = torch.cat(out, dim=0)
    out = out.sum(0) / out.shape[0]
    return out # shape: (res, res, 77)

def aggregate_attention_head(attention_store: AttentionStoreHead, token_to_concept: List[str], res: int, is_cross: bool, select: int, prompts: List[str] = [""]):
    """Average attention maps for the selected head locations.
       res other than 8x8 is either down_sampled or up_sampled before averaging.
       Plese consider rescaling all the attention maps into 64x64,
       and changing kernel_size, sigma appropriately."""
    out = []
    attention_maps = attention_store.get_average_attention() # concept:key: List of (batch_size, item_res^2, 77)
    for concept in token_to_concept:
        average_attention = []
        for location in ["down_cross", "mid_cross", "up_cross"]:
            for item in attention_maps[concept][location]:
                item_res = int(math.sqrt(item.shape[1]))
                cross_maps = item.reshape(len(prompts), -1, item_res, item_res, item.shape[-1])[select] # (1, item_res, item_res, 77)
                if item_res == res:
                    average_attention.append(cross_maps)
                elif item_res < res:
                    cross_maps = cross_maps.permute(0, 3, 1, 2) # (1, 77, item_res, item_res)
                    upsampled_cross_maps = nnf.interpolate(cross_maps, size=(res, res), mode='bicubic', align_corners=False)
                    upsampled_cross_maps = upsampled_cross_maps.permute(0, 2, 3, 1) # (1, res, res, 77)
                    average_attention.append(upsampled_cross_maps)
                else:
                    cross_maps = cross_maps.permute(0, 3, 1, 2) # (1, 77, item_res, item_res)
                    # --- prevent space aliasing during downsampling --- #
                    kernel_size_ = (item_res//res)+1
                    sigma_ = 0.3 * ((kernel_size_ - 1) * 0.5 - 1) + 0.8
                    cross_maps = gaussian_blur(cross_maps, kernel_size=(kernel_size_, kernel_size_), sigma=(sigma_, sigma_))
                    # ----------------------------------------------------- #
                    downsampled_cross_maps = nnf.interpolate(cross_maps, size=(res, res), mode='bicubic', align_corners=False)
                    downsampled_cross_maps = downsampled_cross_maps.permute(0, 2, 3, 1) # (1, res, res, 77)
                    average_attention.append(downsampled_cross_maps)
        average_attention = torch.cat(average_attention, dim=0)
        average_attention = average_attention.sum(0) / average_attention.shape[0] # shape: (res, res, 77)
        out.append(average_attention) 
    out = torch.stack(out, dim=0) # shape: (num_tokens_to_alter, res, res, 77)
    return out # shape: (num_tokens_to_alter, res, res, 77)

"""Define AttendAndExcitePipeline"""
class AttendAndExcitePipeline(StableDiffusionPipeline):
    _optional_components = ["safety_checker", "feature_extractor"]

    def _encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt = False, # I have changed this to boolian type
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    ):
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        
        if prompt_embeds is None:
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            # --- delete this code block if you don't need it --- #
            # Check whether the input token length is longer than 77
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1: -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )
            # --------------------------------------------------- #
            
            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None
            
            prompt_embeds = self.text_encoder(
                text_input_ids.to(device),
                attention_mask=attention_mask,
            )[0]
        
        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is False:
                uncond_tokens = [""] * batch_size
            else:
                uncond_tokens = ["ugly, bad anatomy, bad quality, blurry, disconnected limbs, disfigured, disgusting, fused fingers, long neck, low quality, jpeg artifacts, mutated limbs, mutated fingers, poorly drawn face, watermark, username, error, worst quality, pixel, low resolution, disgusting"]
            
            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None
            
            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )[0]
        
        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

        return text_inputs, prompt_embeds
    
    def _compute_max_attention_per_index(self,
                                         attention_maps: torch.Tensor, # (res, res, 77) [or (num_tokens_to_alter, res, res, 77)]
                                         indices_to_alter: List[int],
                                         smooth_attentions: bool = False,
                                         sigma: float = 0.5,
                                         kernel_size: int = 3,
                                         normalize_before_eot: bool = False # I have changed the name of this variable
                                         ) -> List[torch.Tensor]:
        """Computes the maximum attention value for each of the tokens we wish to alter."""
        # --------------------------------- #
        prompt = self.prompt
        if isinstance(prompt, list):
            prompt = prompt[0]
        if normalize_before_eot:
            last_idx = len(self.tokenizer(prompt)['input_ids']) - 1 
        else:
            last_idx = self.tokenizer.model_max_length # original: -1
        # ---------------------------------- #
        if args.run_standard_attend_excite:
            attention_for_text = attention_maps[:, :, 1:last_idx]
        else:
            attention_for_text = attention_maps[:, :, :, 1:last_idx]
        attention_for_text *= 100 # this values is used in the original Attend-and-Excite github
        attention_for_text = nnf.softmax(attention_for_text, dim=-1)

        # Shift indices since we removed the first token
        indices_to_alter = [index - 1 for index in indices_to_alter]

        # Extract the maximum values
        max_values_list = []
        if args.run_standard_attend_excite:
            for i in indices_to_alter:
                image = attention_for_text[:, :, i] # (res, res)
                if smooth_attentions:
                    smoothing = GaussianSmoothing(channels=1, kernel_size=kernel_size, sigma=sigma, dim=2).to(device)
                    input = nnf.pad(image.unsqueeze(0).unsqueeze(0), [kernel_size//2] * 4, mode='reflect')
                    image = smoothing(input).squeeze(0).squeeze(0)
                max_values_list.append(image.max())
        else:
            for idx, i in enumerate(indices_to_alter):
                image = attention_for_text[idx, :, :, i] # (res, res)
                if smooth_attentions:
                    smoothing = GaussianSmoothing(channels=1, kernel_size=kernel_size, sigma=sigma, dim=2).to(device)
                    input = nnf.pad(image.unsqueeze(0).unsqueeze(0), [kernel_size//2] * 4, mode='reflect')
                    image = smoothing(input).squeeze(0).squeeze(0)
                max_values_list.append(image.max())
        return max_values_list
    
    def _aggregate_and_get_max_attention_per_token(self, attention_store,
                                                   indices_to_alter: List[int],
                                                   token_to_concept: List[str],
                                                   attention_res: int = 16,
                                                   smooth_attentions: bool = False,
                                                   sigma: float = 0.5,
                                                   kernel_size: int = 3,
                                                   normalize_before_eot: bool = False,):
        """Aggregates the attention maps for each token and computes the max activation value for each token to alter."""
        # ---- Whether you use standard Attend-and-Excite or not --- #
        if args.run_standard_attend_excite:
            attention_maps = aggregate_attention(
                attention_store=attention_store,
                res=attention_res,
                from_where=("up", "down", "mid"),
                is_cross=True,
                select=0,
            )
        else:
            attention_maps = torch.stack(attention_store.averaged_maps, dim=0) 
        # ----------------------------------------------------------------------------- #
        max_attention_per_index = self._compute_max_attention_per_index(
            attention_maps=attention_maps,
            indices_to_alter=indices_to_alter,
            smooth_attentions=smooth_attentions,
            sigma=sigma,
            kernel_size=kernel_size,
            normalize_before_eot=normalize_before_eot,
        )
        return max_attention_per_index
    
    @staticmethod
    def _compute_loss(max_attention_per_index: List[torch.Tensor], return_losses: bool = False) -> torch.Tensor:
        """Compute the attend-and-excite loss using the maximum attention value for each token."""
        losses = [max(0, 1. - curr_max) for curr_max in max_attention_per_index]
        loss = max(losses)
        if return_losses:
            return loss, losses
        else:
            return loss
        
    @staticmethod
    def _update_latent(latents: torch.Tensor, loss: torch.Tensor, step_size: float) -> torch.Tensor:
        """Update the latent according to the computed loss."""
        grad_cond = torch.autograd.grad(loss.requires_grad_(True), [latents], retain_graph=True)[0]
        latents = latents - step_size * grad_cond
        return latents
    
    def _perform_iterative_refinement_step(self,
                                           latents: torch.Tensor,
                                           indices_to_alter: List[int],
                                           token_to_concept: List[str],
                                           loss: torch.Tensor,
                                           threshold: float,
                                           text_embeddings: torch.Tensor,
                                           text_input,
                                           attention_store,
                                           step_size: float,
                                           t: int,
                                           attention_res: int = 16,
                                           smooth_attentions: bool = True,
                                           sigma: float = 0.5,
                                           kernel_size: int = 3,
                                           max_refinement_steps: int = 20, # 20
                                           normalize_before_eot: bool = False,):
        """Performs the iterative latent refinement introduced in the paper. Here, we continuously update the latent
        code according to our loss objective until the given threshold is reached for all tokens."""
        iteration = 0
        target_loss = max(0, 1. - threshold)
        while loss > target_loss:
            iteration += 1

            latents = latents.clone().detach().requires_grad_(True)
            noise_pred_text = self.unet(latents, t, encoder_hidden_states=text_embeddings[1].unsqueeze(0)).sample
            self.unet.zero_grad()

            # Get max activation value for each subject token
            max_attention_per_index = self._aggregate_and_get_max_attention_per_token(
                attention_store=attention_store,
                indices_to_alter=indices_to_alter,
                token_to_concept=token_to_concept,
                attention_res=attention_res,
                smooth_attentions=smooth_attentions,
                sigma=sigma,
                kernel_size=kernel_size,
                normalize_before_eot=normalize_before_eot,
            )

            loss, losses = self._compute_loss(max_attention_per_index, return_losses=True)

            if loss != 0:
                latents = self._update_latent(latents, loss, step_size)
            
            with torch.no_grad():
                noise_pred_uncond = self.unet(latents, t, encoder_hidden_states=text_embeddings[0].unsqueeze(0)).sample
                noise_pred_text = self.unet(latents, t, encoder_hidden_states=text_embeddings[1].unsqueeze(0)).sample
            
            try:
                low_token = np.argmax([l.item() if type(l) != int else l for l in losses])
            except Exception as e:
                print(e) # catch edge case
                low_token = np.argmax(losses)
            
            low_word = self.tokenizer.decode(text_input.input_ids[0][indices_to_alter[low_token]])
            print(f"\t Try {iteration}. {low_word} has a max attention of {max_attention_per_index[low_token]}")

            if iteration >= max_refinement_steps:
                print(f'\t Exceeded max number of iterations ({max_refinement_steps})! '
                      f'Finished with a max attention of {max_attention_per_index[low_token]}')
                break
        
        # Run one more time but don't compute gradients and update the latents.
        # We just need to compute the new loss - the grad update will occur below
        latents = latents.clone().detach().requires_grad_(True)
        noise_pred_text = self.unet(latents, t, encoder_hidden_states=text_embeddings[1].unsqueeze(0)).sample
        self.unet.zero_grad()

        # Get max activation value for each subject token
        max_attention_per_index = self._aggregate_and_get_max_attention_per_token(
            attention_store=attention_store,
            indices_to_alter=indices_to_alter,
            token_to_concept=token_to_concept,
            attention_res=attention_res,
            smooth_attentions=smooth_attentions,
            sigma=sigma,
            kernel_size=kernel_size,
            normalize_before_eot=normalize_before_eot,
        )
        loss, losses = self._compute_loss(max_attention_per_index, return_losses=True)
        print(f"\t Finished with loss of: {loss}")
        return loss, latents, max_attention_per_index
    
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        attention_store,
        indices_to_alter: List[int],
        token_to_concept: List[str],
        attention_res: int = 16,
        height: Optional[int] = 512, # 768 -> 512
        width: Optional[int] = 512, # 768 -> 512
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: bool = False,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        max_iter_to_alter: Optional[int] = 25,
        run_standard_sd: bool = False,
        thresholds: Optional[dict] = {0: 0.05, 10: 0.5, 20: 0.8},
        scale_factor: int = 20,
        scale_range: Tuple[float, float] = (1., 0.5),
        smooth_attentions: bool = True,
        sigma: float = 0.5,
        kernel_size: int = 3,
    ):
        """run_standard_sd: inference without Attend-and-Excite
           Returns: [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`]"""
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
        )

        # 2. Define call parameters
        self.prompt = prompt
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        
        device = self._execution_device
        # 'guidance_scale = 1' corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_inputs, prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        scale_range = np.linspace(scale_range[0], scale_range[1], len(self.scheduler.timesteps))

        if max_iter_to_alter is None:
            max_iter_to_alter = len(self.scheduler.timesteps)
        
        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                with torch.enable_grad():
                    latents = latents.clone().detach().requires_grad_(True)

                    # Forward pass of denoising with text conditioning
                    noise_pred_text = self.unet(latents, t, encoder_hidden_states=prompt_embeds[1].unsqueeze(0),
                                                cross_attention_kwargs=cross_attention_kwargs).sample
                    self.unet.zero_grad()

                    # Get max activation value for each subject token
                    max_attention_per_index = self._aggregate_and_get_max_attention_per_token(
                        attention_store=attention_store,
                        indices_to_alter=indices_to_alter,
                        token_to_concept=token_to_concept,
                        attention_res=attention_res,
                        smooth_attentions=smooth_attentions,
                        sigma=sigma,
                        kernel_size=kernel_size,
                        normalize_before_eot=False,  # check this; why True for sd_2_1?
                    )

                    if not run_standard_sd:
                        loss = self._compute_loss(max_attention_per_index=max_attention_per_index)

                        # Perform iterative refinement step
                        if i in thresholds.keys() and loss > 1. - thresholds[i]:
                            del noise_pred_text
                            torch.cuda.empty_cache()
                            loss, latents, max_attention_per_index = self._perform_iterative_refinement_step(
                                latents=latents,
                                indices_to_alter=indices_to_alter,
                                token_to_concept=token_to_concept,
                                loss=loss,
                                threshold=thresholds[i],
                                text_embeddings=prompt_embeds,
                                text_input=text_inputs,
                                attention_store=attention_store,
                                step_size=scale_factor * np.sqrt(scale_range[i]),
                                t=t,
                                attention_res=attention_res,
                                smooth_attentions=smooth_attentions,
                                sigma=sigma,
                                kernel_size=kernel_size,
                                normalize_before_eot=False, # check this
                            )
                        
                        # Perform gradient update
                        if i < max_iter_to_alter:
                            loss = self._compute_loss(max_attention_per_index=max_attention_per_index)
                            if loss != 0:
                                latents = self._update_latent(latents=latents, loss=loss,
                                                              step_size=scale_factor * np.sqrt(scale_range[i]))
                            print(f"Iteration {i} | Loss: {loss:0.4f}")
                
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                ).sample

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                
                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # call the callback function, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % callback_steps == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)
        
        # 8. Post-processing
        image = self.decode_latents(latents)
        # image = latent2image(self.vae, latents)

        # 9. Run safety checker
        image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)

        # 10. Convert to PIL
        if output_type == "pil":
            image = self.numpy_to_pil(image)
        
        if not return_dict:
            return (image, has_nsfw_concept)
        
        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)


"""Define functions from ptp_utils, vis_utils"""
def get_image_grid(images: List[Image.Image]) -> Image:
    num_images = len(images)
    cols = int(math.ceil(math.sqrt(num_images)))
    rows = int(math.ceil(num_images / cols))
    width, height = images[0].size
    grid_image = Image.new('RGB', (cols * width, rows * height))
    for i, img in enumerate(images):
        x = i % cols
        y = i // cols
        grid_image.paste(img, (x * width, y * height))
    return grid_image

def register_attention_control(model, controller): 
    def ca_forward(self, place_in_unet, count):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = to_out[0]
        else:
            to_out = self.to_out
        
        def forward(hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None,):
            is_cross = encoder_hidden_states is not None
            layer = count // 2 # layer index

            residual = hidden_states

            if self.spatial_norm is not None:
                hidden_states = self.spatial_norm(hidden_states, temb)
            
            input_ndim = hidden_states.ndim

            if input_ndim == 4:
                batch_size, channel, height, width = hidden_states.shape
                hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
            
            batch_size, sequence_length, _ = (
                hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
            )
            attention_mask = self.prepare_attention_mask(attention_mask, sequence_length, batch_size)

            if self.group_norm is not None:
                hidden_states = self.group_norm(hidden_states.transpos(1, 2)).transpose(1, 2)
            
            query = self.to_q(hidden_states)

            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            elif self.norm_cross:
                encoder_hidden_states = self.norm_encoder_hidden_states(encoder_hidden_states)
            
            key = self.to_k(encoder_hidden_states)
            value = self.to_v(encoder_hidden_states)
            wo_head = key.shape[0]
            
            query = self.head_to_batch_dim(query)
            key = self.head_to_batch_dim(key)
            value = self.head_to_batch_dim(value)
            w_head = key.shape[0]
            num_heads = int(w_head // wo_head)

            attention_probs = self.get_attention_scores(query, key, attention_mask)
            attention_probs = controller(attention_probs, is_cross, place_in_unet, layer, num_heads)

            hidden_states = torch.bmm(attention_probs, value)
            hidden_states = self.batch_to_head_dim(hidden_states)

            # linear proj
            hidden_states = to_out(hidden_states)

            if input_ndim == 4:
                hidden_states = hidden_states.transpose(1, 2).view(batch_size, channel, height, width)
            
            if self.residual_connection:
                hidden_states = hidden_states + residual
            
            hidden_states = hidden_states / self.rescale_output_factor

            return hidden_states
        return forward
    
    class DummyController:
        def __call__(self, *args):
            return args[0]
        
        def __init__(self):
            self.num_att_layers = 0
    
    if controller is None:
        controller = DummyController()

    def register_recr(net_, count, place_in_unet):
        if net_.__class__.__name__ == 'Attention':
            net_.forward = ca_forward(net_, place_in_unet, count)
            return count + 1
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count
    
    att_count = 0
    sub_nets = model.unet.named_children()
    for net in sub_nets:
        if "down" in net[0]:
            att_count += register_recr(net[1], 0, "down")
        elif "up" in net[0]:
            att_count += register_recr(net[1], 0, "up")
        elif "mid" in net[0]:
            att_count += register_recr(net[1], 0, "mid")
    
    controller.num_att_layers = att_count


def get_indices_to_alter(model, prompt: str) -> List[int]:
    token_idx_to_word = {idx: model.tokenizer.decode(t)
                         for idx, t in enumerate(model.tokenizer(prompt)['input_ids'])
                         if 0 < idx < len(model.tokenizer(prompt)['input_ids']) - 1}
    pprint.pprint(token_idx_to_word)
    token_indices = input("Please enter the a comma-separated list indices of the tokens you wish to "
                          "alter (e.g., 2,5): ")
    token_indices = [int(i) for i in token_indices.split(",")]
    print(f"Altering tokens: {[token_idx_to_word[i] for i in token_indices]}")
    return token_indices

def run_on_prompt(prompt: List[str],
                  model: AttendAndExcitePipeline,
                  controller,
                  token_indices: List[int],
                  token_to_concept: List[str],
                  seed: torch.Generator) -> Image.Image:
    if controller is not None:
        register_attention_control(model, controller)
    outputs = model(prompt=prompt,
                    attention_store=controller,
                    indices_to_alter=token_indices,
                    token_to_concept=token_to_concept,
                    attention_res=args.attention_res,
                    guidance_scale=args.guidance_scale,
                    negative_prompt=args.negative_prompt,
                    generator=seed,
                    num_inference_steps=args.n_inference_steps,
                    max_iter_to_alter=args.max_iter_to_alter,
                    run_standard_sd=args.run_standard_sd,
                    thresholds=args.thresholds,
                    scale_factor=args.scale_factor,
                    scale_range=args.scale_range,
                    smooth_attentions=args.smooth_attentions,
                    sigma=args.sigma,
                    kernel_size=args.kernel_size,
                    )
    image = outputs.images[0]
    return image

"""Main function"""
concepts = list(set(args.concepts))
if args.model_version == "sd_v1_4":
    model = AttendAndExcitePipeline.from_pretrained(args.model_card).to(device)
token_indices = get_indices_to_alter(model, args.prompt) if args.token_indices is None else args.token_indices
if len(args.concepts) != len(token_indices):
    raise ValueError("Length of token_to_concept and token_indices must be the same.")

images = []
for seed in args.seeds:
    print(f"Seed: {seed}")
    g = torch.Generator('cuda').manual_seed(seed)
    if args.run_standard_sd:
        output_dir = os.path.join(args.output_path, "standard_sd", f"exp_{args.experiment_num}")
    elif args.run_standard_attend_excite:
        output_dir = os.path.join(args.output_path, f"original_attend_and_excite",
                                  f"exp_{args.experiment_num}")
    else:
        output_dir = os.path.join(args.output_path, f"attend_and_excite-hrv",
                                  f"exp_{args.experiment_num}")
    prompt_output_path = os.path.join(output_dir, args.prompt)
    os.makedirs(prompt_output_path, exist_ok=True)
    image_save_path = os.path.join(prompt_output_path, f"{seed}.png")
    if os.path.exists(image_save_path):
        continue
    
    if args.run_standard_attend_excite:
        controller = AttentionStore()
    else:
        controller = AttentionStoreHead(concepts)
    image = run_on_prompt(prompt=args.prompt,
                          model=model,
                          controller=controller,
                          token_indices=token_indices,
                          token_to_concept=args.concepts,
                          seed=g)

    image.save(image_save_path)
    images.append(image)

# save a grid of results across all seeds
joined_image = get_image_grid(images)
joined_image.save(os.path.join(output_dir, f"{args.prompt}.png"))





    
