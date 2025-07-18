TRAIN_FILE="../../data/train.csv"
VALID_FILE="../../data/valid.csv"
OUTPUT_DIR="../../checkpoint_training/codeT5Plus"
MODEL_NAME="Salesforce/codet5p-220m"
NUM_EPOCHS=8
PUSH_TO_HUB=true
HUB_MODEL_ID="BienKieu/codeT5Plus"

CUDA_VISIBLE_DEVICES=1 python train.py \
  --train_file "$TRAIN_FILE" \
  --valid_file "$VALID_FILE" \
  --output_dir "$OUTPUT_DIR" \
  --init_model "$MODEL_NAME" \
  --num_train_epochs $NUM_EPOCHS \
  $( [ "$PUSH_TO_HUB" = true ] && echo "--push_to_hub --hub_model_id $HUB_MODEL_ID" )

# chmod +x train.sh
# nohup ./train.sh > train.log 2>&1 &
# tail -f train.log
# ps -ef | grep train