#!/bin/bash
SPLIT="llava_gqa_testdev_balanced"

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

run_gqa(){
    CUDA_VISIBLE_DEVICES=$GPU_ID python -m llava.eval.model_vqa_loader \
        --model-path $CKPT \
        --question-file /visiontrim/playground/data/eval/gqa/$SPLIT.jsonl \
        --image-folder /visiontrim/playground/data/eval/gqa/images \
        --answers-file ../data/eval/gqa/answers/$SPLIT/$method/$token_num.jsonl \
        --num-chunks 1 \
        --temperature 0 \
        --conv-mode vicuna_v1 \
        --method $method \
        --dataset-name gqa \
        --DVTS_token_num $token_num \
        --TGVC_token_num $token_complement

    wait
    output_file=../data/eval/gqa/answers/$SPLIT/$method/$token_num.jsonl
    mkdir -p ../data/eval/gqa/answers/llava_gqa_testdev_balanced/$method/$token_num
    python scripts/convert_gqa_for_eval.py --src "$output_file" --dst ../data/eval/gqa/answers/llava_gqa_testdev_balanced/$method/$token_num/testdev_balanced_predictions.json

    cd ../data/gqa/eval
    python eval.py --tier testdev_balanced --method $method --token_num $token_num
}

method=VisionTrim
token_num=$1
token_complement=$2
CKPT=/llm_checkpoints/llava-v1.5-7b
GPU_ID=2
run_gqa $GPU_ID $method $CKPT $token_num $token_complement