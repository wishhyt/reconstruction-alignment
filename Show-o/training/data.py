# coding=utf-8
# Copyright 2024 The HuggingFace, NUS Show Lab.
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

# This file is heavily inspired by https://github.com/mlfoundations/open_clip/blob/main/src/training/data.py

import itertools
import json
import math
import os
import random
import re
from functools import partial
from typing import List, Optional, Union
import torch
from PIL import Image
import io

Image.warnings.simplefilter('error', Image.DecompressionBombWarning)

import webdataset as wds
import yaml
from braceexpand import braceexpand
from torch.utils.data import default_collate
from torchvision import transforms
from transformers import PreTrainedTokenizer
from webdataset.tariterators import (
    base_plus_ext,
    tar_file_expander,
    url_opener,
    valid_sample,
)

person_token = ["a person", "someone", "somebody"]


def replace_person_token(t):
    "Used for CC12M"
    t = re.sub("<person>([,\s]*(and)*[,\s]*<person>)+", " people ", t)
    while "<person>" in t:
        t = t.replace("<person>", f" {random.choices(person_token)} ", 1)
    return t


def filter_keys(key_set):
    def _f(dictionary):
        return {k: v for k, v in dictionary.items() if k in key_set}

    return _f


def group_by_keys_nothrow(data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None):
    """Return function over iterator that groups key, value pairs into samples.

    :param keys: function that splits the key into key and extension (base_plus_ext)
    :param lcase: convert suffixes to lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        fname, value = filesample["fname"], filesample["data"]
        prefix, suffix = keys(fname)
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        # FIXME webdataset version throws if suffix in current_sample, but we have a potential for
        #  this happening in the current LAION400m dataset if a tar ends with same prefix as the next
        #  begins, rare, but can happen since prefix aren't unique across tar files in that dataset
        if current_sample is None or prefix != current_sample["__key__"] or suffix in current_sample:
            if valid_sample(current_sample):
                yield current_sample
            current_sample = dict(__key__=prefix, __url__=filesample["__url__"])
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if valid_sample(current_sample):
        yield current_sample


def tarfile_to_samples_nothrow(src, handler=wds.warn_and_continue):
    # NOTE this is a re-impl of the webdataset impl with group_by_keys that doesn't throw
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler)
    samples = group_by_keys_nothrow(files, handler=handler)
    return samples


def image_transform(sample, resolution=256):
    image = sample["images"]
    image = transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC)(image)
    image = transforms.CenterCrop((resolution, resolution))(image)
    image = transforms.ToTensor()(image)
    image = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)(image)
    sample["images"] = image
    return sample


def remove_prefix(caption):
    caption = caption.replace('The image features ', '').replace('The image presents ', '').replace(
        "The image you've sent is, ", '').replace("In the center of the image, ", '').replace(
        "The image showcases ", '').replace("The image is ", '').replace(
        "The image captures ", '').replace("In the given image ", '').replace(
        "The image portrays ", '').replace("In the image, ", '').replace("In this image, we see ", '').replace(
        "The image depicts ", '').replace("This is ", '').replace("In this image, ", '').replace(
        "This image captures ", '')

    return caption

from PIL import Image, ImageFilter, ImageDraw
import cv2
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as T
import random

class MidjourneyDataset(Dataset):
    def __init__(
        self,
        base_dataset,
        image_size: int = 512,
        gen_prompt_type = None
    ):
        self.base_dataset = base_dataset
        self.image_size = image_size
        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.gen_prompt_type = gen_prompt_type

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        target = 0
        item = self.base_dataset[idx]
        # item['image'] is base64 encoded image
        image = Image.open(io.BytesIO(item["image"])).convert("RGB")
        # image = item["image"].convert("RGB")
        image = image.resize(
            (self.image_size, self.image_size)
        ).convert("RGB")

        description = item["prompt"]
        
        if self.gen_prompt_type is not None:
            if random.random() < 0.5:
                description = item["llava"]
        
        return {
            "images": self.normalize(self.to_tensor(image)),
            "input_ids": description,
        }
        

class Subject200KDataset(Dataset):
    def __init__(
        self,
        base_dataset,
        condition_size: int = 512,
        target_size: int = 512,
        image_size: int = 512,
        padding: int = 12,
        condition_type: str = "subject",
        drop_text_prob: float = 0,
        drop_image_prob: float = 0,
        return_pil_image: bool = False,
    ):
        self.base_dataset = base_dataset
        self.condition_size = condition_size
        self.target_size = target_size
        self.image_size = image_size
        self.padding = padding
        self.condition_type = condition_type
        self.drop_text_prob = drop_text_prob
        self.drop_image_prob = drop_image_prob
        self.return_pil_image = return_pil_image
        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])


    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        # If target is 0, left image is target, right image is condition
        target = 0
        item = self.base_dataset[idx]

        # Crop the image to target and condition
        image = item["image"]

        # left_img = image.crop(
        #     (
        #         self.padding,
        #         self.padding,
        #         self.image_size + self.padding,
        #         self.image_size + self.padding,
        #     )
        # )
        left_img = image.crop(
            (
                0,
                0,
                self.image_size ,
                self.image_size ,
            )
        )
        right_img = image.crop(
            (
                self.image_size + self.padding * 2,
                self.padding,
                self.image_size * 2 + self.padding * 2,
                self.image_size + self.padding,
            )
        )

        # Get the target and condition image
        target_image, condition_img = (
            (left_img, right_img) if target == 0 else (right_img, left_img)
        )

        # Resize the image
        condition_img = condition_img.resize(
            (self.condition_size, self.condition_size)
        ).convert("RGB")
        target_image = target_image.resize(
            (self.target_size, self.target_size)
        ).convert("RGB")

        # Get the description
        description = item["description"][
            "description_0" if target == 0 else "description_1"
        ]

        # Randomly drop text or image
        drop_text = random.random() < self.drop_text_prob
        drop_image = random.random() < self.drop_image_prob
        if drop_text:
            description = ""
        if drop_image:
            condition_img = Image.new(
                "RGB", (self.condition_size, self.condition_size), (0, 0, 0)
            )
        # print("subject:", item["description"]['category'])
        return {
            "image": self.normalize(self.to_tensor(target_image)),
            "condition": self.normalize(self.to_tensor(condition_img)),
            "condition_type": self.condition_type,
            "description": description,
            "subject": item["description"]['item'],
            # 16 is the downscale factor of the image
            "position_delta": np.array([0, -self.condition_size // 16]),
            **({"pil_image": image} if self.return_pil_image else {}),
        }
        
class Text2ImageDataset:
    def __init__(
            self,
            train_shards_path_or_url: Union[str, List[str]],
            tokenizer: PreTrainedTokenizer,
            max_seq_length: int,
            num_train_examples: int,
            per_gpu_batch_size: int,
            global_batch_size: int,
            num_workers: int,
            resolution: int = 256,
            shuffle_buffer_size: int = 1000,
            pin_memory: bool = False,
            persistent_workers: bool = False,
            external_caption_path: Optional[str] = '',
            external_journeydb_caption_path: Optional[str] = '',
            external_laion12m_caption_path: Optional[str] = '',
            external_cc12m_caption_path: Optional[str] = '',
            is_captioning: bool = False,
            add_caption_prompt: bool = False,
            long_caption: bool = True,
            wds_seed: int = 42,
    ):
        if f"{train_shards_path_or_url}.yaml" in os.listdir('./configs'):
            with open(f"./configs/{train_shards_path_or_url}.yaml") as f:
                train_shards_path_or_url = yaml.safe_load(f)
        self.long_caption = long_caption
        self.external_caption_path = external_caption_path
        self.external_journeydb_caption_path = external_journeydb_caption_path
        self.external_laion12m_caption_path = external_laion12m_caption_path
        self.external_cc12m_caption_path = external_cc12m_caption_path
        self.is_captioning = is_captioning
        self.add_caption_prompt = add_caption_prompt
        if self.add_caption_prompt:
            with open("./training/questions.json") as f:
                self.caption_prompt = json.load(f)
                self.caption_prompt = ['USER: \n' + prompt + ' ASSISTANT:' for prompt in self.caption_prompt]
        else:
            self.caption_prompt = None

        if external_journeydb_caption_path != '':
            with open(external_journeydb_caption_path) as file:
                self.journeydb_caption = json.load(file)
                
            print(f"Loaded JourneyDB captions from: {external_journeydb_caption_path} with {len(self.journeydb_caption)} entries.")
        else:
            self.journeydb_caption = None

        def tokenize(text):
            if tokenizer is not None:
                text = replace_person_token(text)
                input_ids = tokenizer(
                    text, max_length=max_seq_length, padding="max_length", truncation=True, return_tensors="pt"
                ).input_ids
                return input_ids[0]
            else:
                return text

        if not isinstance(train_shards_path_or_url, str):
            train_shards_path_or_url = [list(braceexpand(urls)) for urls in train_shards_path_or_url]
            # flatten list using itertools
            train_shards_path_or_url = list(itertools.chain.from_iterable(train_shards_path_or_url))

        # if external_caption_path != '':
        processing_pipeline = [
            wds.decode("pil", handler=wds.ignore_and_continue),
            wds.map(self.load_external_caption, handler=wds.ignore_and_continue),
            wds.rename(
                images="jpg;png;jpeg;webp",
                input_ids="text;txt;caption",
                handler=wds.warn_and_continue,
            ),
            wds.map(filter_keys(set(["images", "input_ids"]))),
            wds.map(partial(image_transform, resolution=resolution), handler=wds.warn_and_continue),
            wds.map_dict(
                input_ids=tokenize,
                handler=wds.warn_and_continue,
            ),
        ]
        # else:
        #     processing_pipeline = [
        #         wds.decode("pil", handler=wds.ignore_and_continue),
        #         wds.rename(
        #             images="jpg;png;jpeg;webp",
        #             input_ids="text;txt;caption",
        #             handler=wds.warn_and_continue,
        #         ),
        #         wds.map(filter_keys(set(["images", "input_ids"]))),
        #         wds.map(partial(image_transform, resolution=resolution), handler=wds.warn_and_continue),
        #         wds.map_dict(
        #             input_ids=tokenize,
        #             handler=wds.warn_and_continue,
        #         ),
        #     ]

        pipeline = [
            wds.ResampledShards(train_shards_path_or_url, deterministic=True),
            tarfile_to_samples_nothrow,
            wds.shuffle(shuffle_buffer_size, rng=random.Random(wds_seed)),
            *processing_pipeline,
            wds.batched(per_gpu_batch_size, partial=False, collation_fn=default_collate),
        ]

        num_batches = math.ceil(num_train_examples / global_batch_size)
        num_worker_batches = math.ceil(num_train_examples / (global_batch_size * num_workers))  # per dataloader worker
        num_batches = num_worker_batches * num_workers
        num_samples = num_batches * global_batch_size

        # each worker is iterating over this
        self._train_dataset = wds.DataPipeline(*pipeline).with_epoch(num_worker_batches)
        
        def seed_worker(worker_id: int):
            worker_seed = worker_id
            np.random.seed(worker_seed)
            random.seed(worker_seed)
            print(f"Worker {worker_id} seed: {worker_seed}")
        
        self._train_dataloader = wds.WebLoader(
            self._train_dataset,
            batch_size=None,
            shuffle=False,
            num_workers=num_workers,
            # num_workers=0,
            pin_memory=pin_memory,
            worker_init_fn=seed_worker,
            persistent_workers=persistent_workers,
            # persistent_workers=0,
        )
        # add meta-data to dataloader instance for convenience
        self._train_dataloader.num_batches = num_batches
        self._train_dataloader.num_samples = num_samples

    def load_external_caption(self, sample):

        if 'txt' not in sample.keys():
            sample['txt'] = ''

        if 'SA1B' in sample['__key__']:
            assert False, "No SA1B"
            captionf = f"{self.external_caption_path}/{sample['__key__'].split('/')[-1]}.txt"
            if os.path.exists(captionf):
                with open(captionf, "r") as reader:
                    captions = reader.readlines()[0].replace('\n', '')
            else:
                captions = ""

            # for captioning
            if self.is_captioning:
                if self.add_caption_prompt is not None:
                    prompt = random.sample(self.caption_prompt, 1)[0]
                    sample['txt'] = prompt + ' ' + captions
                else:
                    sample['txt'] = captions
            # for generation
            else:
                # randomly choose short and long captions
                if random.random() < 0.5:
                    sample['txt'] = captions.split('.')[0]
                else:
                    sample['txt'] = captions

                sample['txt'] = remove_prefix(sample['txt'])

            return sample

        elif 'laion' in sample['__url__']:
            assert False, "No laion"
            captionf = f"{self.external_laion12m_caption_path}/{sample['__url__'].split('/')[-1].split('.')[0]}/{sample['__key__']}.caption"
            if os.path.exists(captionf):
                with open(captionf, "r") as reader:
                    captions = reader.readlines()[0].replace('\n', '')
            else:
                captions = ""

            # for captioning
            if self.is_captioning:
                if self.add_caption_prompt is not None:
                    prompt = random.sample(self.caption_prompt, 1)[0]
                    sample['txt'] = prompt + ' ' + captions
                else:
                    sample['txt'] = captions
            # for generation
            else:
                # randomly choose short and long captions
                if random.random() < 0.5:
                    sample['txt'] = captions.split('.')[0]
                else:
                    sample['txt'] = captions

                sample['txt'] = remove_prefix(sample['txt'])

            return sample

        elif 'cc12m' in sample['__url__']:
            assert False, "No cc12m"
            captionf = f"{self.external_cc12m_caption_path}/{sample['__url__'].split('/')[-1].split('.')[0]}/{sample['__key__'].split('/')[-1]}.caption"
            if os.path.exists(captionf):
                with open(captionf, "r") as reader:
                    captions = reader.readlines()[0].replace('\n', '')
            else:
                captions = ""

            # for captioning
            if self.is_captioning:
                if self.add_caption_prompt is not None:
                    prompt = random.sample(self.caption_prompt, 1)[0]
                    sample['txt'] = prompt + ' ' + captions
                else:
                    sample['txt'] = captions
            # for generation
            else:
                # randomly choose short and long captions
                if random.random() < 0.5:
                    sample['txt'] = captions.split('.')[0]
                else:
                    sample['txt'] = captions
                sample['txt'] = remove_prefix(sample['txt'])

            return sample

        elif self.journeydb_caption is not None and sample['__key__'] in self.journeydb_caption:
            # sample['txt'] = random.sample(self.journeydb_caption[sample['__key__']], 1)[0]
            # return sample
            sample['txt'] = self.journeydb_caption[sample['__key__']]
            return sample

        else:
            print(
                f"Warning: No external caption found for {sample['__key__']}. Using default empty string."
            )
            # print(sample)
            return sample

    @property
    def train_dataset(self):
        return self._train_dataset

    @property
    def train_dataloader(self):
        return self._train_dataloader


if __name__ == '__main__':
    pass
