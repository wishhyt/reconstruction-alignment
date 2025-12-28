# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

# Memory-efficient training script for BAGEL
# This script is optimized for GPUs with limited memory

# ===========================================
# GPU Selection (指定GPU入口)
# ===========================================
# Set CUDA_VISIBLE_DEVICES to specify which GPUs to use
# Examples:
#   export CUDA_VISIBLE_DEVICES=0        # Use only GPU 0
#   export CUDA_VISIBLE_DEVICES=0,1      # Use GPU 0 and 1
#   export CUDA_VISIBLE_DEVICES=2,3,4,5  # Use GPU 2, 3, 4, 5
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

# Number of GPUs (should match the number of GPUs in CUDA_VISIBLE_DEVICES)
GPUS=${GPUS:-8}

# ===========================================
# Training Configuration
# ===========================================
# Replace these variables with your own paths
num_nodes=${num_nodes:-1}
node_rank=${node_rank:-0}
master_addr=${master_addr:-localhost}
master_port=${master_port:-12345}

# Model paths
model_path=${model_path:-"ckpt/BAGEL-7B-MoT"}
llm_path=${llm_path:-""}
vae_path=${vae_path:-""}
vit_path=${vit_path:-""}
resume_from=${resume_from:-"ckpt/BAGEL-7B-MoT"}

# Output paths
output_path=${output_path:-"./results"}
ckpt_path=${ckpt_path:-"./checkpoints"}

# ===========================================
# Memory Optimization Settings
# ===========================================
# cpu_offload: Offload model parameters to CPU (saves most memory, slower training)
CPU_OFFLOAD=${CPU_OFFLOAD:-False}

# disable_ema: Disable EMA model (saves ~50% model memory)
DISABLE_EMA=${DISABLE_EMA:-False}

# vae_encode_batch_size: Encode images in mini-batches (0 = encode all at once)
# Recommended: 2-8 for limited memory
VAE_ENCODE_BATCH_SIZE=${VAE_ENCODE_BATCH_SIZE:-0}

# empty_cache_every: Clear CUDA cache every N steps (0 = disabled)
# Recommended: 10-50 for fragmentation issues
EMPTY_CACHE_EVERY=${EMPTY_CACHE_EVERY:-0}

# sharding_strategy: HYBRID_SHARD (default) or FULL_SHARD (more memory efficient)
SHARDING_STRATEGY=${SHARDING_STRATEGY:-HYBRID_SHARD}

# max_latent_size: Reduce for lower memory (default 64, use 32 for limited memory)
MAX_LATENT_SIZE=${MAX_LATENT_SIZE:-64}

export PYTHONPATH=.

echo "=== Memory-Efficient BAGEL Training ==="
echo "GPUs: $CUDA_VISIBLE_DEVICES (count: $GPUS)"
echo "CPU Offload: $CPU_OFFLOAD"
echo "Disable EMA: $DISABLE_EMA"
echo "VAE Encode Batch Size: $VAE_ENCODE_BATCH_SIZE"
echo "Empty Cache Every: $EMPTY_CACHE_EVERY steps"
echo "Sharding Strategy: $SHARDING_STRATEGY"
echo "Max Latent Size: $MAX_LATENT_SIZE"
echo "========================================"

torchrun \
  --nnodes=$num_nodes \
  --node_rank=$node_rank \
  --nproc_per_node=$GPUS \
  --master_addr=$master_addr \
  --master_port=$master_port \
  train/pretrain_unified_navit.py \
  --model_path $model_path \
  --dataset_config_file ./data/configs/example.yaml \
  --layer_module Qwen2MoTDecoderLayer \
  --max_latent_size $MAX_LATENT_SIZE \
  --freeze_vae True \
  --freeze_vit True \
  --freeze_und True \
  --finetune_from_ema True \
  --resume_from $resume_from \
  --results_dir $output_path \
  --checkpoint_dir $ckpt_path \
  --use_flex True \
  --cpu_offload $CPU_OFFLOAD \
  --disable_ema $DISABLE_EMA \
  --vae_encode_batch_size $VAE_ENCODE_BATCH_SIZE \
  --empty_cache_every $EMPTY_CACHE_EVERY \
  --sharding_strategy $SHARDING_STRATEGY \
  "$@"
