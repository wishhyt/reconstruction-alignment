import os
import random
import numpy as np
import torch

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

os.environ["TOKENIZERS_PARALLELISM"] = "true"
from PIL import Image
from tqdm import tqdm, trange
import json
# import wandb
from models import Showo, MAGVITv2, get_mask_chedule
from training.prompting_utils import UniversalPrompting, create_attention_mask_predict_next
from training.utils import get_config, flatten_omega_conf, image_transform
from transformers import AutoTokenizer
import torch.nn.functional as F
from datasets import load_dataset

def get_vq_model_class(model_type):
    if model_type == "magvitv2":
        return MAGVITv2
    else:
        raise ValueError(f"model_type {model_type} not supported.")

if __name__ == '__main__':

    config = get_config()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(config.model.showo.llm_model_path, padding_side="left")

    uni_prompting = UniversalPrompting(tokenizer, max_text_len=config.dataset.preprocessing.max_seq_length,
                                       special_tokens=("<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>", "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>"),
                                       ignore_id=-100, cond_dropout_prob=config.training.cond_dropout_prob)

    vq_model = get_vq_model_class(config.model.vq_model.type)
    vq_model = vq_model.from_pretrained(config.model.vq_model.vq_model_name).to(device)
    vq_model.requires_grad_(False)
    vq_model.eval()

    model = Showo.from_pretrained(config.pretrained).to(device)
    model.eval()

    mask_token_id = model.config.mask_token_id

    # load from users passed arguments
    if config.get("validation_prompts_file", None) is not None:
        config.dataset.params.validation_prompts_file = config.validation_prompts_file
    config.training.batch_size = config.batch_size
    config.training.guidance_scale = config.guidance_scale
    config.training.generation_timesteps = config.generation_timesteps

    with open('../Benchmark/dpgbench/prompts.json', 'r') as f:
        dataset = json.load(f)

    l = config.get("l", 0)
    r = config.get("r", len(dataset))
    
    for key, prompt in dataset.items():
        index = list(dataset.keys()).index(key)
        if index < l or index >= r:
            continue 
               
        num_images = 4
        print(f"Prompt ({index}/{len(dataset)}, key={key}): '{prompt}'")
        os.makedirs(config.get("outdir"), exist_ok=True)
        
        
        # if config.get("use_template", False):
        #     prompt = 'USER: \n Describe the image: ' + prompt + ' ASSISTANT:'
        
        for img_idx in range(num_images):
            set_seed(img_idx)
            out_path = os.path.join(config.get("outdir"), f"{key.split('.')[-2]}_{img_idx}.jpg")
            if os.path.exists(out_path):
                print(f"Sample {out_path} already exists, skipping...")
                continue

            prompts = [prompt]
            
            image_tokens = torch.ones((len(prompts), config.model.showo.num_vq_tokens),
                                    dtype=torch.long, device=device) * mask_token_id

            input_ids, _ = uni_prompting((prompts, image_tokens), 't2i_gen')
            
            if config.training.guidance_scale > 0:
                uncond_input_ids, _ = uni_prompting(([''] * len(prompts), image_tokens), 't2i_gen')
                attention_mask = create_attention_mask_predict_next(torch.cat([input_ids, uncond_input_ids], dim=0),
                                                                    pad_id=int(uni_prompting.sptids_dict['<|pad|>']),
                                                                    soi_id=int(uni_prompting.sptids_dict['<|soi|>']),
                                                                    eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']),
                                                                    rm_pad_in_image=True)
            else:
                attention_mask = create_attention_mask_predict_next(input_ids,
                                                                    pad_id=int(uni_prompting.sptids_dict['<|pad|>']),
                                                                    soi_id=int(uni_prompting.sptids_dict['<|soi|>']),
                                                                    eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']),
                                                                    rm_pad_in_image=True)
                uncond_input_ids = None

            if config.get("mask_schedule", None) is not None:
                schedule = config.mask_schedule.schedule
                args = config.mask_schedule.get("params", {})
                mask_schedule = get_mask_chedule(schedule, **args)
            else:
                mask_schedule = get_mask_chedule(config.training.get("mask_schedule", "cosine"))

            with torch.no_grad():
                gen_token_ids = model.t2i_generate(
                    input_ids=input_ids,
                    uncond_input_ids=uncond_input_ids,
                    attention_mask=attention_mask,
                    guidance_scale=config.training.guidance_scale,
                    temperature=config.training.get("generation_temperature", 1.0),
                    timesteps=config.training.generation_timesteps,
                    noise_schedule=mask_schedule,
                    noise_type=config.training.get("noise_type", "mask"),
                    seq_len=config.model.showo.num_vq_tokens,
                    uni_prompting=uni_prompting,
                    config=config,
                )

            gen_token_ids = torch.clamp(gen_token_ids, max=config.model.showo.codebook_size - 1, min=0)
            images = vq_model.decode_code(gen_token_ids)

            images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
            images *= 255.0
            images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
            image = Image.fromarray(images[0])
            image.save(out_path, format='JPEG')
            
            print(f"Generated image {img_idx+1}/4 for prompt key '{key}', saved to {out_path}")

    print("Done")