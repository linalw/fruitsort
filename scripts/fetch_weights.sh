#!/usr/bin/env bash
# 断点续传下载权重/数据集到 data/checkpoints 或指定目录。
# 用法:
#   bash scripts/fetch_weights.sh <url> [文件名] [目标子目录]
# 例:
#   bash scripts/fetch_weights.sh https://huggingface.co/lerobot/act_aloha_sim_transfer_cube_human/resolve/main/model.safetensors
#   bash scripts/fetch_weights.sh <url> vla-3b.safetensors checkpoints
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
URL="${1:?用法: fetch_weights.sh <url> [文件名] [子目录]}"
NAME="${2:-$(basename "${URL%%\?*}")}"
SUB="${3:-checkpoints}"
DEST="$ROOT/data/$SUB"
mkdir -p "$DEST"

echo "[fetch] $URL"
echo "[fetch] → $DEST/$NAME"
curl -L --fail --retry 5 --retry-delay 5 -C - \
  --progress-bar -o "$DEST/$NAME" "$URL"

echo "[fetch] 完成: $(du -h "$DEST/$NAME" | cut -f1)  $DEST/$NAME"
