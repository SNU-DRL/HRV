"""Import libraries"""
from typing import Optional, Union, Tuple, List, Callable, Dict
import torch
from diffusers import StableDiffusionPipeline, DiffusionPipeline
import os
import pickle
import argparse
from tqdm import tqdm


# Arguments
def get_args_parser():
    parser = argparse.ArgumentParser("Description_to_text_embeddings", add_help=False)

    # Dataset parameters
    parser.add_argument("--gpu_num", default=0, type=int)
    parser.add_argument('--description', type=str, default='340_final_text_descriptions')
    parser.add_argument('--model_card', type=str, default="CompVis/stable-diffusion-v1-4") # "stabilityai/stable-diffusion-xl-base-1.0"
    return parser

args = get_args_parser()
args = args.parse_args()

# Access information
LOW_RESOURCE = False
NUM_DIFFUSION_STEPS = 50
GUIDANCE_SCALE = 7.5
MAX_NUM_WORDS = 77
MODEL_CARD = args.model_card 
device = torch.device(f"cuda:{args.gpu_num}") if torch.cuda.is_available() else torch.device("cpu")

if MODEL_CARD == "CompVis/stable-diffusion-v1-4":
    ldm_stable = StableDiffusionPipeline.from_pretrained(MODEL_CARD).to(device)
    args.model_version = "sd_v1_4"
elif MODEL_CARD == "stabilityai/stable-diffusion-xl-base-1.0":
    ldm_stable = DiffusionPipeline.from_pretrained(MODEL_CARD).to(device)
    args.model_version = "sd_xl_base_1_0"
tokenizer = ldm_stable.tokenizer

"""This code load description txt file"""
cwd = os.getcwd()
file_path = os.path.join(cwd, "descriptions", f"{args.description}.txt")

# Load a description txt file
with open(file_path, 'r') as f:
    lines = f.readlines()

# Replace '\n' to ''
for i, line in enumerate(lines):
    lines[i] = line.replace('\n', '')


"""Cacluate text embeddings for descriptions and save them"""
# Calculate text embeddings
model = ldm_stable
chunk_size = 32
len_tokens = [] # a list, whose elements are token lengths of concept-words
autocast = torch.cuda.amp.autocast

if args.model_version == "sd_xl_base_1_0":
    with torch.no_grad(), autocast():
        desc_embeddings = []
        for i in tqdm(range(0, len(lines), chunk_size)):
            texts = lines[i:i+chunk_size]
            (prompt_embeds, _, _, _) = ldm_stable.encode_prompt(texts, num_images_per_prompt=1, do_classifier_free_guidance=True)
            desc_embeddings.append(prompt_embeds)
        desc_embeddings = torch.concatenate(desc_embeddings, dim=0).detach().cpu()
        for phrase in lines:
            tokens = model.tokenizer.encode(phrase)
            len_tokens.append(len(tokens))
else:
    with torch.no_grad(), autocast():
        desc_embeddings = []
        for i in tqdm(range(0, len(lines), chunk_size)):
            texts = lines[i:i+chunk_size]
            texts = model.tokenizer(texts, padding="max_length", max_length=model.tokenizer.model_max_length, truncation=True, return_tensors="pt",).to(model.device)
            desc_embeddings.append(model.text_encoder(texts.input_ids.to(model.device))[0])
        desc_embeddings = torch.concatenate(desc_embeddings, dim=0).detach().cpu()
        for phrase in lines:
            tokens = model.tokenizer.encode(phrase)
            len_tokens.append(len(tokens))

# Save the results; (1) text embeddings for concept-words, (2) token lengths for concept-words
os.makedirs(os.path.join(cwd, "word_lists"), exist_ok=True)
torch.save(desc_embeddings, os.path.join(cwd, "word_lists", f"text_embedding_{args.description}_{args.model_version}.pt"))

with open(os.path.join(cwd, "word_lists", f"len_tokens_{args.description}_{args.model_version}.pkl"), 'wb') as f:
    pickle.dump(len_tokens, f)