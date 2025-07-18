TRAIN_FILE="../../self_improvement/output_self/train_SI.csv"
VALID_FILE="../../data/valid.csv"
OUTPUT_DIR="../../checkpoint_training/codeT5Plus_Contrastive_SI"
MODEL_NAME="BienKieu/codeT5Plus_Contrastive"
NUM_EPOCHS=4
PUSH_TO_HUB=true
HUB_MODEL_ID="BienKieu/codeT5Plus_Contrastive_SI"

CUDA_VISIBLE_DEVICES=2 python train.py \
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