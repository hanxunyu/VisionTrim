#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT="llava-v1.5-7b"
method=VisionTrim
token_num=$1
token_complement=$2
GPU_ID=1

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=$GPU_ID python -W ignore -m llava.eval.model_vqa_loader \
        --model-path /llm_checkpoints/${CKPT} \
        --question-file /visiontrim/playground/data/eval/vqav2/llava_vqav2_mscoco_test-dev2015.jsonl \
        --image-folder /visiontrim/playground/data/eval/vqav2/test2015 \
        --answers-file ../data/eval/vqav2/answers/$method/$token_num/vision_encoder+${CHUNKS}_${IDX}.jsonl \
        --num-chunks ${CHUNKS} \
        --chunk-idx ${IDX} \
        --temperature 0 \
        --conv-mode vicuna_v1 \
        --method $method \
        --dataset-name vqav2 \
        --DVTS_token_num $token_num \
        --TGVC_token_num $token_complement
done

wait

VQAV2DIR="../data/eval/vqav2"
output_file=${VQAV2DIR}/answers/$method/$token_num/vision_encoder+merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ../data/eval/vqav2/answers/$method/$token_num/vision_encoder+${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python scripts/convert_vqav2_for_submission.py \
    --dir ${VQAV2DIR} \
    --src answers/$method/$token_num/vision_encoder+merge.jsonl \
    --dst answers_upload/$method/$token_num/vision_encoder+$token_num+$method-upload.json

