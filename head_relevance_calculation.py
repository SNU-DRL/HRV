"""Import libraries"""
from typing import Optional, Union, Tuple, List, Callable, Dict
import torch
from diffusers import StableDiffusionPipeline
import numpy as np
import abc
from PIL import Image
import cv2
from IPython.display import display
from tqdm import tqdm
import os
import pickle
import argparse

# Arguments
def get_args_parser():
    parser = argparse.ArgumentParser("Head relevance calculation", add_help=False)

    # Dataset parameters
    parser.add_argument("--gpu_num", default=0, type=int)
    parser.add_argument("--subset_running", action='store_true', help="Use the subset_running flag to run the code on a subset of the data, which can be useful when using multiple GPUs.")
    parser.add_argument("--numerator", default=1, type=int, help="Which subset of the data to run the code on") 
    parser.add_argument("--denominator", default=5, type=int, help="The total number of subsets")
    parser.add_argument('--prompt', type=str, default='merged', help="Generation prompts for random image generation, e.g., 'merged' for 2100 prompts, 'imagenet' for 1000 prompts")
    parser.add_argument('--description', type=str, default='340_final_text_descriptions', help="The filename for the set of concept-words")
    parser.add_argument('--model_card', type=str, default="CompVis/stable-diffusion-v1-4") 
    parser.add_argument('--negative_prompt', action='store_true')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--num_words_per_concept', type=int, default=10)
    parser.add_argument('--num_concepts', type=int, default=34)
    return parser

args = get_args_parser()
args = args.parse_args()

# Access information
NUM_DIFFUSION_STEPS = 50
GUIDANCE_SCALE = 7.5
MAX_NUM_WORDS = 77
MODEL_CARD = args.model_card 
device = torch.device("cuda:{}".format(args.gpu_num)) if torch.cuda.is_available() else torch.device("cpu")
if MODEL_CARD == "CompVis/stable-diffusion-v1-4":
    args.model_version = "sd_v1_4"
    ldm_stable = StableDiffusionPipeline.from_pretrained(MODEL_CARD).to(device)
tokenizer = ldm_stable.tokenizer


"""Attention Controllers"""
class AttentionControl(abc.ABC):

    def step_callback(self, x_t):
        return x_t
    
    def between_steps(self):
        return
    
    @property
    def num_uncond_att_layers(self):
        return 0
    
    @abc.abstractmethod
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            h = attn.shape[0] 
            attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet) 
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

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        return attn

class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}
    
    def forward(self, attn ,is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 32 ** 2:  # avoid memory overhead
            self.step_store[key].append(attn)
        return attn
    
    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()
    
    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
        return average_attention
    
    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}
    
    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}


class SimilarityStore(AttentionControl):
    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        self.similarity_store[self.cur_step][key].append(attn.detach().cpu()) # avoid memory overhead
        return attn

    def __call__(self, attn, is_cross: bool, place_in_unet: str): 
        if is_cross:
            if self.cur_att_layer >= self.num_uncond_att_layers:
                self.forward(attn, is_cross, place_in_unet) 
        else:
            pass
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return None

    def return_similarity(self):
        return self.similarity_store

    def __init__(self, num_inference_steps=50, model_card=None):
        super(SimilarityStore, self).__init__()
        self.model_card = model_card
        self.total_steps = num_inference_steps
        self.similarity_store = [self.get_empty_store() for _ in range(self.total_steps + 1)] # You have to change (self.total_steps + 1) to (self.total_steps) for SD v2.x

    def reset(self):
        super(SimilarityStore, self).reset()
        self.similarity_store = [self.get_empty_store() for _ in range(self.total_steps + 1)] # You have to change (self.total_steps + 1) to (self.total_steps) for SD v2.x

## Define functions
def text_under_image(image: np.ndarray, text: str, text_color: Tuple[int, int, int] = (0, 0, 0)):
    h, w, c = image.shape
    offset = int(h * .2)
    img = np.ones((h + offset, w, c), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    img[:h] = image
    textsize = cv2.getTextSize(text, font, 1, 2)[0]
    text_x, text_y = (w - textsize[0]) // 2, h + offset - textsize[1] // 2
    cv2.putText(img, text, (text_x, text_y), font, 1, text_color, 2)
    return img


def view_images(images, num_rows=1, offset_ratio=0.02):
    if type(images) is list:
        num_empty = num_rows - len(images) % num_rows
    elif images.ndim == 4:
        num_empty = num_rows - images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0
    
    empty_image = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_image] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = np.ones((h * num_rows + offset * (num_rows - 1), w * num_cols + offset * (num_cols - 1), c),
                     dtype=np.uint8) * 255
    for i in range(num_rows):
        for j in range(num_cols):
            image_[i * (h + offset): i * (h + offset) + h, j * (w + offset): j * (w + offset) + w] = images[
                i * num_cols + j]
    pil_img = Image.fromarray(image_)
    display(pil_img)


def diffusion_step(model, controller, latents, context, t, guidance_scale):
    latents_input = torch.cat([latents] * 2)
    noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)["sample"] 
    noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
    latents = model.scheduler.step(noise_pred, t, latents)["prev_sample"]
    latents = controller.step_callback(latents)
    return latents


def latent2image(vae, latents):
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents)['sample']
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).astype(np.uint8)
    return image


def init_latent(latent, model, height, width, generator, batch_size):
    if latent is None:
        latent = torch.randn(
            (1, model.unet.config.in_channels, height // 8, width // 8),
            generator=generator
        )
    latents = latent.expand(batch_size, model.unet.config.in_channels, height // 8, width // 8).to(model.device)
    return latent, latents


@torch.no_grad()
def text2image_ldm_stable(
    model,
    prompt: List[str],
    controller,
    desc_embeddings, 
    len_tokens: List[int], 
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    generator: Optional[torch.Generator] = None,
    latent: Optional[torch.FloatTensor] = None,
    negative_prompt: bool = False, # We do not use negative prompts
): 
    
    register_attention_control(model, controller, desc_embeddings, len_tokens) 
    height = width = 512
    batch_size = len(prompt)

    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    max_length = text_input.input_ids.shape[-1]
    if negative_prompt:
        uncond_input = model.tokenizer(
            ["ugly, bad anatomy, bad quality, blurry, disconnected limbs, disfigured, disgusting, fused fingers, long neck, low quality, jpeg artifacts, mutated limbs, mutated fingers, poorly drawn face, watermark, username, error, worst quality, pixel, low resolution, disgusting"] * batch_size, 
            padding="max_length", max_length=max_length, return_tensors="pt"
        )
    else:
        uncond_input = model.tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
    uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]

    context = [uncond_embeddings, text_embeddings]
    context = torch.cat(context)
    latent, latents = init_latent(latent, model, height, width, generator, batch_size)

    model.scheduler.set_timesteps(num_inference_steps)
    for t in model.scheduler.timesteps:
        latents = diffusion_step(model, controller, latents, context, t , guidance_scale)

    image = latent2image(model.vae, latents)

    return image, latent


def register_attention_control(model, controller, desc_embeddings, len_tokens): 
    def ca_forward(self, place_in_unet, count):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = to_out[0]
        else:
            to_out = self.to_out
        
        def forward(hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None,):
            is_cross = encoder_hidden_states is not None
            layer = (count) // 2 # layer index

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
                hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
            
            query = self.to_q(hidden_states)

            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            elif self.norm_cross:
                encoder_hidden_states = self.norm_encoder_hidden_states(encoder_hidden_states)

            key = self.to_k(encoder_hidden_states) 
            value = self.to_v(encoder_hidden_states)
            wo_head = key.shape[0] # 2*batch_size

            query = self.head_to_batch_dim(query)
            key = self.head_to_batch_dim(key) 
            value = self.head_to_batch_dim(value)
            w_head = key.shape[0] # 2*batch_size*h
            num_head = int(w_head // wo_head)

            if is_cross:
                num_desc = desc_embeddings.shape[0]
                random_integers = torch.randint(low=0, high=args.num_words_per_concept, size=(args.num_concepts,))
                index_list = torch.arange(0, num_desc, args.num_words_per_concept) + random_integers
                sampled_desc_embeddings = desc_embeddings[index_list] # (num_concept, 77, hid)
                sampled_len_tokens = np.array(len_tokens)[index_list] # (num_concept, )
                aggregate_desc = []
                for idx, desc in enumerate(sampled_desc_embeddings):
                    len_desc = sampled_len_tokens[idx] - 2
                    if idx == 0:
                        aggregate_desc.append(desc[:len_desc + 1]) # this code includes <SOT> token before the softmax operation, which is not necessary (Note that we will apply argmax over the concept-words)
                        # aggregate_desc.append(desc[1:len_desc + 1]) # this code does not include <SOT> token before the softmax operation
                    elif idx == len(sampled_desc_embeddings) - 1:
                        # aggregate_desc.append(desc[1:]) # this code includes <EOT> tokens before the softmax operation, which is not necessary
                        aggregate_desc.append(desc[1:len_desc + 1]) # this code does not include <EOT> token before the softmax operation
                    else:
                        aggregate_desc.append(desc[1:len_desc + 1])
                aggregate_desc = torch.concat(aggregate_desc, dim=0)
                sampled_desc_embeddings = aggregate_desc.unsqueeze(0) # (1, 77, hid)

                desc_key = self.to_k(sampled_desc_embeddings) 
                desc_key = self.head_to_batch_dim(desc_key) # (1*h, 77, hid)
                desc_attention_probs = self.get_attention_scores(query[query.shape[0] // 2:], desc_key, attention_mask) # (h, res*res, 77)
                desc_attention_probs = desc_attention_probs.mean(dim=1) # (h, 77)
                aggregate_sim = []
                cnt = 0
                for idx, len_desc in enumerate(sampled_len_tokens):
                    len_desc -= 2
                    if idx == 0:
                        aggregate_sim.append(desc_attention_probs[:, 1:len_desc +1].mean(dim=1, keepdim=True))
                        cnt += len_desc + 1 # If <SOT> token is included
                        # cnt += len_desc # If <SOT> token is not included
                    else:
                        aggregate_sim.append(desc_attention_probs[:, cnt: cnt + len_desc].mean(dim=1, keepdim=True))
                        cnt += len_desc
                result = torch.concat(aggregate_sim, dim=-1) # (h, num_concept)
                max_indices = result.argmax(dim=-1).unsqueeze(-1)
                result = torch.zeros_like(result).scatter_(1, max_indices, 1) # (h, num_concept)
            else:
                result = None
                pass

            attention_probs = self.get_attention_scores(query, key, attention_mask) # shape: [2*batch*h, res*res, 77]
            # --------------------------------------------------- #
            controller(result, is_cross, place_in_unet)
            # --------------------------------------------------- #
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
        if net_.__class__.__name__ == "Attention":
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
    


def run_only(prompts, controller, desc_embeddings, len_tokens, latent=None, generator=None):
    images, x_t = text2image_ldm_stable(ldm_stable, prompts, controller, desc_embeddings, len_tokens, latent=latent, num_inference_steps=NUM_DIFFUSION_STEPS, guidance_scale=GUIDANCE_SCALE, generator=generator, negative_prompt=args.negative_prompt) 
    return images, x_t


"""Load text embeddings of text directions"""
cwd = os.getcwd()

# Load the tensor
desc_embeddings = torch.load(os.path.join(cwd, "word_lists", f"text_embedding_{args.description}_{args.model_version}.pt"))
desc_embeddings = desc_embeddings.to(device)

# Load the list of integers
with open(os.path.join(cwd, "word_lists", f"len_tokens_{args.description}_{args.model_version}.pkl"), 'rb') as f:
    len_tokens = pickle.load(f)


"""Main code"""
# Prepare your prompt files
prompt_file_path = os.path.join(os.getcwd(), "prompt", f"prompt_list_{args.prompt}.txt")
epochs = range(1, args.epochs + 1)
for epoch in epochs:
    with open(prompt_file_path, 'r') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        lines[i] = line.replace('\n', '')

    # Run diffusion process line by line
    total_data = []
    controller = SimilarityStore(num_inference_steps=NUM_DIFFUSION_STEPS, model_card=MODEL_CARD) 
    num_lines = len(lines)
    if args.subset_running:
        lines = lines[(args.numerator-1)*num_lines // args.denominator:args.numerator*num_lines // args.denominator]

    # image_save_folder = os.path.join(cwd, "generated_images", f"{args.prompt}_neg_prompt_{args.negative_prompt}")
    # os.makedirs(image_save_folder, exist_ok=True)

    g_cpu = torch.Generator().manual_seed(22)
    for i, line in enumerate(tqdm(lines)):
        prompts = [line]
        image, _ = run_only(prompts, controller, desc_embeddings, len_tokens, latent=None, generator=g_cpu)
        # if i % 10 == 0: # Save the image every 10 iterations
        #     image_save_path = os.path.join(image_save_folder, f"{(args.numerator-1)*num_lines // args.denominator + i}.png")
        #     imageio.imsave(image_save_path, image.squeeze())
        total_data.append(controller.return_similarity())
        torch.cuda.empty_cache() # disallocate unnecessary memory
        controller.reset()


    """Save the similarity scores"""
    print("Saving the data...")
    os.makedirs(os.path.join(cwd, "results"), exist_ok=True)

    if args.subset_running:
        # You have to concatenate along 0-th dim after loading these subfiles (refer to the "do_the_analysis.ipynb")
        save_path = os.path.join(cwd, "results", f"{args.prompt}_{args.description}_ba_epoch_{epoch}_neg_prompt_{args.negative_prompt}_{args.numerator}_{args.denominator}_{args.model_version}.pt")
    else:
        # You don't have to concatenate
        save_path = os.path.join(cwd, "results", f"{args.prompt}_{args.description}_ba_epoch_{epoch}_neg_prompt_{args.negative_prompt}_{args.model_version}.pt")
    torch.save(total_data, save_path)