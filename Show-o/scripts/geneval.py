'''
python3 geneval.py config=configs/showo_demo_512x512.yaml \
                batch_size=1 mode=t2i\
                guidance_scale=5 generation_timesteps=50 \
'''


# coding=utf-8
# Copyright 2024 NUS Show Lab.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

def get_vq_model_class(model_type):
    if model_type == "magvitv2":
        return MAGVITv2
    else:
        raise ValueError(f"model_type {model_type} not supported.")

if __name__ == '__main__':

    config = get_config()
    set_seed()

    # resume_wandb_run = config.wandb.resume
    # run_id = config.wandb.get("run_id", None)
    # if run_id is None:
    #     resume_wandb_run = False
    #     run_id = wandb.util.generate_id()
    #     config.wandb.run_id = run_id

    # wandb_config = {k: v for k, v in flatten_omega_conf(config, resolve=True)}

    # wandb.init(
    #     project="demo",
    #     name=config.experiment.name + '_t2i' + f'_{config.mode}',
    #     config=wandb_config,
    # )
    
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
    # load from users passed arguments


    with open('../Benchmark/geneval/prompts/evaluation_metadata.jsonl') as fp:
        metadatas = [json.loads(line) for line in fp] 

    l = config.get("l", 0)
    r = config.get("r", len(metadatas))
    for index, metadata in tqdm(enumerate(metadatas)):
        if index < l or index >= r:
            # Skip indices outside the specified range
            continue
        
        
        outpath = os.path.join(config.get("outdir") + ('_2' if  config.get("use_template_2", False) else ''), f"{index:0>5}") 
        os.makedirs(outpath, exist_ok=True)

        # if config.get("use_template", False):
        #     prompt = 'USER: \n Describe the image: ' + metadata['prompt'] + ' ASSISTANT:'
        # elif config.get("use_template_2", False):
        #     prompt = 'USER: \n Imagine what the scene would look like, and describe it: ' + metadata['prompt'] + ' ASSISTANT:'
        # else:
        prompt = metadata['prompt']
        
        batch_size = 1
        print(f"Prompt ({index: >3}/{len(metadatas)}): '{prompt}'")

        sample_path = os.path.join(outpath, "samples")
        os.makedirs(sample_path, exist_ok=True)
        with open(os.path.join(outpath, "metadata.jsonl"), "w") as fp:
            json.dump(metadata, fp)

        sample_count = 0
            
            
        for n in trange((config.get("num_samples", 12) + batch_size - 1) // batch_size, desc="Sampling"):
            set_seed(n)
            out_path = os.path.join(sample_path, f"{sample_count:05}.png")
            if os.path.exists(out_path):
                print(f"Sample {out_path} already exists, skipping...")
                sample_count += 1
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
            image.save(os.path.join(sample_path, f"{sample_count:05}.png"))
            sample_count += 1
            
    print("Done")