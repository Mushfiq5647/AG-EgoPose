#!/usr/bin/env bash

set -euo pipefail

START_EPOCH="${1:-20}"
HEATMAP_PATH="${2:-utils/trained_heatmaps/bce_combined/heatmap_best.ckpt}"
SCENEEGO_DIR="${3:-sceneego-finetune}"
EGOPW_DIR="${3:-utils/trained_egopwtrain_bce_seq64}"
SUMMARY_FILE="test_epoch_summary_from_${START_EPOCH}.txt"

if ! [[ "$START_EPOCH" =~ ^[0-9]+$ ]]; then
  echo "Start epoch must be an integer, got: $START_EPOCH" >&2
  exit 1
fi

if [[ ! -f "$HEATMAP_PATH" ]]; then
  echo "Heatmap checkpoint not found: $HEATMAP_PATH" >&2
  exit 1
fi

mapfile -t encoder_files < <(find "$SCENEEGO_DIR" -maxdepth 1 -type f -name 'encoder-*.ckpt' | sort)

if [[ "${#encoder_files[@]}" -eq 0 ]]; then
  echo "No encoder checkpoints found in $SCENEEGO_DIR" >&2
  exit 1
fi

epoch_count=0
sum_mpjpe=0
sum_pa_mpjpe=0

printf "Epoch\tMPJPE\tPA-MPJPE\n" > "$SUMMARY_FILE"

for encoder_path in "${encoder_files[@]}"; do
  encoder_file="$(basename "$encoder_path")"
  epoch="${encoder_file#encoder-}"
  epoch="${epoch%.ckpt}"

  if (( 10#$epoch < 10#$START_EPOCH )); then
    continue
  fi

  decoder_path="$SCENEEGO_DIR/pose-decoder-${epoch}.ckpt"
  heatmap_embedding_path="$SCENEEGO_DIR/heatmap_embedding-${epoch}.ckpt"
  spatial_transformer_path="$SCENEEGO_DIR/spatial_transformer-${epoch}.ckpt"

  if [[ ! -f "$decoder_path" || ! -f "$heatmap_embedding_path" || ! -f "$spatial_transformer_path" ]]; then
    echo "Skipping epoch $epoch because one or more matching checkpoints are missing." >&2
    continue
  fi

  echo "Running test for epoch $epoch"
  run_output="$(python test.py \
    --encoder_path "$encoder_path" \
    --decoder_path "$decoder_path" \
    --heatmap_trained_path "$HEATMAP_PATH" \
    --heatmap_path "$heatmap_embedding_path" \
    --spatial_transformer_path "$spatial_transformer_path")"

  printf "%s\n" "$run_output"

  mpjpe="$(printf "%s\n" "$run_output" | awk -F': ' '/Average MPJPE/ {print $2}' | tail -n 1)"
  pa_mpjpe="$(printf "%s\n" "$run_output" | awk -F': ' '/Average PA-MPJPE/ {print $2}' | tail -n 1)"

  if [[ -z "$mpjpe" || -z "$pa_mpjpe" ]]; then
    echo "Failed to parse metrics for epoch $epoch" >&2
    exit 1
  fi

  printf "%s\t%s\t%s\n" "$epoch" "$mpjpe" "$pa_mpjpe" | tee -a "$SUMMARY_FILE"

  sum_mpjpe="$(awk -v a="$sum_mpjpe" -v b="$mpjpe" 'BEGIN {printf "%.6f", a + b}')"
  sum_pa_mpjpe="$(awk -v a="$sum_pa_mpjpe" -v b="$pa_mpjpe" 'BEGIN {printf "%.6f", a + b}')"
  epoch_count=$((epoch_count + 1))
done

if (( epoch_count == 0 )); then
  echo "No matching epochs found from $START_EPOCH onward." >&2
  exit 1
fi

overall_mpjpe="$(awk -v total="$sum_mpjpe" -v count="$epoch_count" 'BEGIN {printf "%.4f", total / count}')"
overall_pa_mpjpe="$(awk -v total="$sum_pa_mpjpe" -v count="$epoch_count" 'BEGIN {printf "%.4f", total / count}')"

{
  printf "\nOverall mean from epoch %s onward\n" "$START_EPOCH"
  printf "Average MPJPE across epochs: %s\n" "$overall_mpjpe"
  printf "Average PA-MPJPE across epochs: %s\n" "$overall_pa_mpjpe"
} | tee -a "$SUMMARY_FILE"
