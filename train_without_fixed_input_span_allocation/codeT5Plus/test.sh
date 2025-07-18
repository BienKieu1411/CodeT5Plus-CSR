MODEL="BienKieu/codeT5Plus_new"
DATA_DIR="../../data_merged"
BASE_OUTPUT="../../results/train_without_fixed_input_span_allocation/codeTPlus_new"

# Chọn 1 trong 4 kiểu rerank:
# none / textrank / cross-encoder / hybrid
RERANK_METHOD="textrank"    # ← Đổi chỗ này để test từng loại

ALPHA=0.7  # Trọng số cross-encoder trong hybrid, chỉ áp dụng nếu RERANK_METHOD=hybrid

if [ "$RERANK_METHOD" = "none" ]; then
  OUTPUT_DIR="$BASE_OUTPUT/none"
  USE_TEXTRANK=""
  RERANK_ARG=""
  ALPHA_ARG=""
else
  OUTPUT_DIR="$BASE_OUTPUT/$RERANK_METHOD"
  USE_TEXTRANK="--use_textrank"
  RERANK_ARG="--rerank_method $RERANK_METHOD"

  if [ "$RERANK_METHOD" = "hybrid" ]; then
    ALPHA_ARG="--alpha $ALPHA"
  else
    ALPHA_ARG=""
  fi
fi

mkdir -p "$OUTPUT_DIR"

CUDA_VISIBLE_DEVICES=1 python test.py \
  --model $MODEL \
  --output_dir $OUTPUT_DIR \
  --data_dir $DATA_DIR \
  --num_candidates 30 \
  $USE_TEXTRANK \
  $RERANK_ARG \
  $ALPHA_ARG
  
# chmod +x test.sh
# nohup ./test.sh > test.log 2>&1 &