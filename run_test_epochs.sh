#!/usr/bin/env bash
# Usage:
#   ./run_test_epochs.sh [START_EPOCH] [MODEL_DIR] [ABLATION_FLAG]
#   MODEL_DIR: relative (resolves under /data/My_Backup/ag-egopose-ckpt/) or absolute
# Examples:
#   ./run_test_epochs.sh 20 egopw-ablation-skip-sjt --skip_sjt
#   ./run_test_epochs.sh 20 egopw-ablation-skip-temporal --skip_temporal
#   ./run_test_epochs.sh 20 /data/My_Backup/sceneego-finetune

set -euo pipefail

CKPT_BASE="/data/My_Backup/ag-egopose-ckpt"
HEATMAP_PATH="${CKPT_BASE}/trained_heatmaps/bce_combined/heatmap_best.ckpt"

START_EPOCH="${1:-20}"
MODEL_DIR_ARG="${2:-}"
ABLATION_FLAG="${3:-}"  # Optional: e.g. --skip_temporal, --skip_spatial, --skip_sjt, --skip_mcjq

# Resolve MODEL_DIR: absolute paths used as-is, relative resolved under CKPT_BASE
if [[ "$MODEL_DIR_ARG" = /* ]]; then
  MODEL_DIR="$MODEL_DIR_ARG"
else
  MODEL_DIR="${CKPT_BASE}/${MODEL_DIR_ARG}"
fi
MODEL_NAME="$(basename "$MODEL_DIR")"
SUMMARY_FILE="test_epoch_summary_from_${START_EPOCH}--${MODEL_NAME}.txt"

if ! [[ "$START_EPOCH" =~ ^[0-9]+$ ]]; then
  echo "Start epoch must be an integer, got: $START_EPOCH" >&2
  exit 1
fi

if [[ -z "$MODEL_DIR_ARG" ]]; then
  echo "Usage: $0 START_EPOCH MODEL_DIR [ABLATION_FLAG]" >&2
  exit 1
fi

if [[ ! -f "$HEATMAP_PATH" ]]; then
  echo "Heatmap checkpoint not found: $HEATMAP_PATH" >&2
  exit 1
fi

echo "Using model dir : $MODEL_DIR"
echo "Using heatmap   : $HEATMAP_PATH"

# Look for checkpoint files (encoder if available, otherwise pose-decoder for ablations)
if compgen -G "$MODEL_DIR/encoder-*.ckpt" > /dev/null 2>&1; then
  mapfile -t checkpoint_files < <(find "$MODEL_DIR" -maxdepth 1 -type f -name 'encoder-*.ckpt' | sort)
  checkpoint_type="encoder"
elif compgen -G "$MODEL_DIR/pose-decoder-*.ckpt" > /dev/null 2>&1; then
  mapfile -t checkpoint_files < <(find "$MODEL_DIR" -maxdepth 1 -type f -name 'pose-decoder-*.ckpt' | sort)
  checkpoint_type="pose-decoder"
else
  echo "No encoder or pose-decoder checkpoints found in $MODEL_DIR" >&2
  exit 1
fi

if [[ "${#checkpoint_files[@]}" -eq 0 ]]; then
  echo "No checkpoints found in $MODEL_DIR" >&2
  exit 1
fi

epoch_count=0
sum_mpjpe=0
sum_pa_mpjpe=0

printf "Epoch\tMPJPE\tPA-MPJPE\n" > "$SUMMARY_FILE"

for checkpoint_path in "${checkpoint_files[@]}"; do
  checkpoint_file="$(basename "$checkpoint_path")"

  # Extract epoch number based on checkpoint type
  if [[ "$checkpoint_type" == "encoder" ]]; then
    epoch="${checkpoint_file#encoder-}"
  elif [[ "$checkpoint_type" == "pose-decoder" ]]; then
    epoch="${checkpoint_file#pose-decoder-}"
  fi
  epoch="${epoch%.ckpt}"

  if (( 10#$epoch < 10#$START_EPOCH )); then
    continue
  fi

  encoder_path="$MODEL_DIR/encoder-${epoch}.ckpt"
  decoder_path="$MODEL_DIR/pose-decoder-${epoch}.ckpt"
  heatmap_embedding_path="$MODEL_DIR/heatmap_embedding-${epoch}.ckpt"
  spatial_transformer_path="$MODEL_DIR/spatial_transformer-${epoch}.ckpt"

  # Check file existence based on ablation flags
  missing_files=()
  [[ ! -f "$decoder_path" ]] && missing_files+=("pose-decoder")

  # Only check spatial files if not doing --skip_spatial ablation
  if [[ ! "$ABLATION_FLAG" =~ skip_spatial ]]; then
    [[ ! -f "$heatmap_embedding_path" ]] && missing_files+=("heatmap_embedding")
    # Skip SJT check if --skip_sjt ablation
    if [[ ! "$ABLATION_FLAG" =~ skip_sjt ]]; then
      [[ ! -f "$spatial_transformer_path" ]] && missing_files+=("spatial_transformer")
    fi
  fi

  # Only check encoder if not doing --skip_temporal ablation
  if [[ ! "$ABLATION_FLAG" =~ skip_temporal ]] && [[ "$checkpoint_type" == "encoder" ]]; then
    [[ ! -f "$encoder_path" ]] && missing_files+=("encoder")
  fi

  if [[ ${#missing_files[@]} -gt 0 ]]; then
    echo "Skipping epoch $epoch because missing: ${missing_files[*]}" >&2
    continue
  fi

  echo "Running test for epoch $epoch${ABLATION_FLAG:+ with $ABLATION_FLAG}"
  run_output="$(python test.py \
    --encoder_path "$encoder_path" \
    --decoder_path "$decoder_path" \
    --heatmap_trained_path "$HEATMAP_PATH" \
    --heatmap_path "$heatmap_embedding_path" \
    --spatial_transformer_path "$spatial_transformer_path" \
    $ABLATION_FLAG)"

  printf "%s\n" "$run_output"

  mpjpe="$(printf "%s\n" "$run_output" | awk '/^MPJPE/ {print $NF}')"
  pa_mpjpe="$(printf "%s\n" "$run_output" | awk '/^PA-MPJPE/ {print $NF}')"

  if [[ -z "$mpjpe" || -z "$pa_mpjpe" ]]; then
    echo "Failed to parse metrics for epoch $epoch" >&2
    echo "Debug: mpjpe='$mpjpe', pa_mpjpe='$pa_mpjpe'" >&2
    continue
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
