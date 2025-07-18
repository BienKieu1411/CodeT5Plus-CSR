CUDA_VISIBLE_DEVICES=0 python train.py \
    --train_file '../../data_KP/train_phase2_wKey_phrases_SI.csv' \
    --valid_file '../../data_merged/valid_merged_data.csv' \
    --text_column "text" \
    --summary_column "title" \
    --output_dir "./checkpoint" \
    --model_name_or_path "BienKieu/codeT5Plus_Contrastive_KP" \
    --max_source_length 512 \
    --max_target_length 64 \
    --num_train_epochs 4 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --ignore_pad_token_for_loss \
    --source_prefix "" \
    --push_to_hub \
    --hub_model_id "BienKieu/codeT5Plus_Contrastive_KP_SI"

# nohup ./train.sh > train.log 2>&1 &