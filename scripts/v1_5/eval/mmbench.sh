#!/bin/bash
run_mmbench() {
    SPLIT="mmbench_dev_20230712"
    CUDA_VISIBLE_DEVICES=$GPU_ID bash -c "
        python -m llava.eval.model_vqa_mmbench \
            --model-path $CKPT \
            --question-file /visiontrim/playground/data/eval/mmbench/mmbench_dev_20230712.tsv \
            --answers-file ../data/eval/mmbench/answers/$SPLIT/$method/$token_num.jsonl \
            --single-pred-prompt \
            --temperature 0 \
            --method $method \
            --DVTS_token_num $token_num \
            --TGVC_token_num $token_complement \
            --dataset-name mmbench \
            --conv-mode vicuna_v1

        mkdir -p ../data/eval/mmbench/answers_upload/$SPLIT/$method

        python scripts/convert_mmbench_for_submission.py \
            --annotation-file /visiontrim/playground/data/eval/mmbench/mmbench_dev_20230712.tsv \
            --result-dir ../data/eval/mmbench/answers/$SPLIT/$method \
            --upload-dir ../data/eval/mmbench/answers_upload/$SPLIT/$method \
            --experiment $token_num
    "
}
NAME=mmbench
method=VisionTrim
CKPT=/llm_checkpoints/llava-v1.5-7b
GPU_ID=0
token_num=$1
token_complement=$2

run_mmbench $GPU_ID $method $CKPT $token_num $token_complement
