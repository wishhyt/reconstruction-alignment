# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import os
from pathlib import Path

import torch
import yaml
from PIL import Image

from data.data_utils import pil_img2rgb, patchify
from data.transforms import ImageTransform
from eval.vlm.utils import load_model_and_tokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Encode a multimodal dataset with BAGEL encoders, store embeddings, and compute means."
        )
    )
    parser.add_argument("--model_path", required=True, help="Path to BAGEL model weights.")
    parser.add_argument("--jsonl_path", required=True, help="Path to JSONL dataset file.")
    parser.add_argument("--image_root", required=True, help="Root directory for image files.")
    parser.add_argument("--output_dir", required=True, help="Directory to save embeddings.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for encoding.")
    parser.add_argument(
        "--config_path",
        default="data/configs/example.yaml",
        help="Path to data config for image transforms.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device to use (e.g., cuda, cuda:0, cpu). Defaults to cuda if available.",
    )
    return parser.parse_args()


def load_image_transform(config_path: str) -> ImageTransform:
    with open(config_path, "r") as f:
        data_config = yaml.safe_load(f)

    if "vlm_sft" in data_config:
        transform_args = data_config["vlm_sft"]["image_transform_args"]
    elif "reconstruction" in data_config and "vit_image_transform_args" in data_config["reconstruction"]:
        transform_args = data_config["reconstruction"]["vit_image_transform_args"]
    else:
        transform_args = {
            "image_stride": 14,
            "max_image_size": 224,
            "min_image_size": 224,
        }

    return ImageTransform(
        max_image_size=transform_args["max_image_size"],
        min_image_size=transform_args["min_image_size"],
        image_stride=transform_args["image_stride"],
        max_pixels=transform_args.get("max_pixels", 14 * 14 * 9 * 1024),
    )


def extract_text(sample: dict) -> str | None:
    for key in ("text", "caption", "prompt", "question"):
        if key in sample and isinstance(sample[key], str):
            return sample[key]
    if "conversations" in sample and isinstance(sample["conversations"], list):
        texts = []
        for item in sample["conversations"]:
            value = item.get("value") if isinstance(item, dict) else None
            if isinstance(value, str):
                texts.append(value)
        if texts:
            return " ".join(texts)
    return None


def extract_image_path(sample: dict) -> str | None:
    if "image" in sample:
        image_value = sample["image"]
    elif "images" in sample:
        image_value = sample["images"]
    else:
        return None

    if isinstance(image_value, list) and image_value:
        return image_value[0]
    if isinstance(image_value, str):
        return image_value
    return None


def encode_texts(model, tokenizer, new_token_ids, texts, device):
    pad_token_id = tokenizer.pad_token_id
    token_ids = []
    lengths = []
    for text in texts:
        ids = tokenizer.encode(text)
        ids = [new_token_ids["bos_token_id"]] + ids + [new_token_ids["eos_token_id"]]
        token_ids.append(ids)
        lengths.append(len(ids))

    max_len = max(lengths)
    input_ids = torch.full((len(texts), max_len), pad_token_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((len(texts), max_len), dtype=torch.long, device=device)
    for idx, ids in enumerate(token_ids):
        input_ids[idx, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
        attention_mask[idx, : len(ids)] = 1

    with torch.no_grad():
        outputs = model.language_model.model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state

    mask = attention_mask.unsqueeze(-1).float()
    summed = (hidden_states * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1)
    return summed / counts


def encode_images(model, transform, images, device):
    image_tensors = []
    seqlens = []
    position_ids = []
    vit_tokens = []
    for image in images:
        image_tensor = transform(image)
        image_tensors.append(image_tensor)
        height, width = image_tensor.shape[1:]
        pos_ids = model.get_flattened_position_ids(
            height,
            width,
            model.vit_patch_size,
            max_num_patches_per_side=model.vit_max_num_patch_per_side,
        )
        tokens = patchify(image_tensor, model.vit_patch_size)
        seqlens.append(tokens.shape[0])
        position_ids.append(pos_ids)
        vit_tokens.append(tokens)

    packed_vit_tokens = torch.cat(vit_tokens, dim=0).to(device)
    packed_position_ids = torch.cat(position_ids, dim=0).to(device)
    vit_token_seqlens = torch.tensor(seqlens, dtype=torch.int, device=device)
    cu_seqlens = torch.nn.functional.pad(torch.cumsum(vit_token_seqlens, dim=0), (1, 0)).to(torch.int32)
    max_seqlen = torch.max(vit_token_seqlens).item()

    with torch.no_grad():
        packed_vit_token_embed = model.vit_model(
            packed_pixel_values=packed_vit_tokens,
            packed_flattened_position_ids=packed_position_ids,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        packed_vit_token_embed = model.connector(packed_vit_token_embed)
        pos_emb = model.vit_pos_embed(packed_position_ids)
        packed_vit_token_embed = packed_vit_token_embed + pos_emb

    split_embeddings = torch.split(packed_vit_token_embed, seqlens, dim=0)
    pooled = [emb.mean(dim=0) for emb in split_embeddings]
    return torch.stack(pooled, dim=0)


def iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main():
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer, new_token_ids = load_model_and_tokenizer(args)
    model = model.to(device).eval()
    transform = load_image_transform(args.config_path)

    text_embeddings = []
    image_embeddings = []
    text_sum = torch.zeros(model.hidden_size, device=device)
    image_sum = torch.zeros(model.hidden_size, device=device)
    total_count = 0

    batch_texts = []
    batch_images = []

    for sample in iter_jsonl(args.jsonl_path):
        text = extract_text(sample)
        image_path = extract_image_path(sample)
        if not text or not image_path:
            continue

        if not os.path.isabs(image_path):
            image_path = os.path.join(args.image_root, image_path)

        if not os.path.exists(image_path):
            continue

        image = pil_img2rgb(Image.open(image_path))
        batch_texts.append(text)
        batch_images.append(image)

        if len(batch_texts) >= args.batch_size:
            text_batch = encode_texts(model, tokenizer, new_token_ids, batch_texts, device)
            image_batch = encode_images(model, transform, batch_images, device)

            text_embeddings.append(text_batch.cpu())
            image_embeddings.append(image_batch.cpu())
            text_sum += text_batch.sum(dim=0)
            image_sum += image_batch.sum(dim=0)
            total_count += text_batch.shape[0]
            batch_texts = []
            batch_images = []

    if batch_texts:
        text_batch = encode_texts(model, tokenizer, new_token_ids, batch_texts, device)
        image_batch = encode_images(model, transform, batch_images, device)
        text_embeddings.append(text_batch.cpu())
        image_embeddings.append(image_batch.cpu())
        text_sum += text_batch.sum(dim=0)
        image_sum += image_batch.sum(dim=0)
        total_count += text_batch.shape[0]

    if total_count == 0:
        raise RuntimeError("No valid samples found for encoding.")

    text_embeddings = torch.cat(text_embeddings, dim=0)
    image_embeddings = torch.cat(image_embeddings, dim=0)
    text_mean = (text_sum / total_count).cpu()
    image_mean = (image_sum / total_count).cpu()

    output_path = output_dir / "bagel_embeddings.pt"
    torch.save(
        {
            "text_embeddings": text_embeddings,
            "image_embeddings": image_embeddings,
            "text_mean": text_mean,
            "image_mean": image_mean,
            "count": total_count,
        },
        output_path,
    )

    print(f"Saved embeddings to {output_path}")


if __name__ == "__main__":
    main()
