#!/usr/bin/env bash
set -euo pipefail


echo "🚀 Starting training..."
# 필요시 CLI 인자 전달 가능 (환경에 맞게 조정)
python /workspace/model.py \
  --layout /workspace/fab_oht_layout_updated.json \
  --data_dirs /workspace/datasets_pivot \
  --epochs 50 \
  --batch_size 32 \
  --seq_len 12 \
  --horizons 6 30 60 \
  --hidden 128 \
  --blocks 3 \
  --lr 1e-3 \
  # --dropout 0.1 \
  2>&1 | tee /workspace/train.log


echo "✅ Training finished. Logs saved to /workspace/train.log"
