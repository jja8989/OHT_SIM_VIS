#!/usr/bin/env bash
set -euo pipefail


echo "🚀 Starting training..."

  python model_4ch_2.py \
  --mode train \
  --layout fab_oht_layout_updated.json \
  --dataset_dirs ./datasets/dynamic ./datasets/pivot\
  --epochs 50 --blocks 2 --hidden 32 \
  --batch_size 16 --seq_len 18 --horizons 1 3 6 \
  --num_workers 0 --amp --K_spatial 1


echo "✅ Training finished. Logs saved to /workspace/train.log"
