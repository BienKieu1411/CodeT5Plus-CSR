MODEL="BienKieu/codeT5Plus_Contrastive_KP_SI"
DATA_DIR="../../data_KP"
BASE_OUTPUT="../../results/codeT5Plus_Contrastive_KP_SI"

# Chọn 1 trong 4 kiểu rerank:
# none / textrank / cross-encoder / hybrid
RERANK_METHOD="none"    # ← Đổi chỗ này để test từng loại

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

CUDA_VISIBLE_DEVICES=0 python test.py \
  --model $MODEL \
  --output_dir $OUTPUT_DIR \
  --data_dir $DATA_DIR \
  --num_candidates 30 \
  $USE_TEXTRANK \
  $RERANK_ARG \
  $ALPHA_ARG
  
# chmod +x test.sh
# nohup ./test.sh > test.log 2>&1 &