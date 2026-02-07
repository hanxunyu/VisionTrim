#!/bin/bash
run_mmbench_cn() {
    SPLIT="mmbench_dev_cn_20231003"
    CUDA_VISIBLE_DEVICES=$GPU_ID bash -c "
        python -m llava.eval.model_vqa_mmbench \
            --model-path $CKPT \
            --question-file ../data/mmbench/mmbench_dev_cn_20231003.tsv \
            --answers-file ../data/eval/mmbench/answers/$SPLIT/$method/$token_num.jsonl \
            --lang cn \
            --single-pred-prompt \
            --temperature 0 \
            --method $method \
            --DVTS_token_num $token_num \
            --TGVC_token_num $token_complement \
            --dataset-name mmbench_cn \
            --conv-mode vicuna_v1

        mkdir -p ../data/eval/mmbench/answers_upload/$SPLIT/$method/$token_num

        python scripts/convert_mmbench_for_submission.py \
            --annotation-file ../data/mmbench/mmbench_dev_cn_20231003.tsv \
            --result-dir ../data/eval/mmbench/answers/$SPLIT/$method/$token_num \
            --upload-dir ../data/eval/mmbench/answers_upload/$SPLIT/$method/$token_num \
            --experiment $NAME
    "
}

method=VisionTrim
CKPT=/llm_checkpoints/llava-v1.5-7b
GPU_ID=5,6
token_num=$1
token_complement=$2

run_mmbench_cn $GPU_ID $method $CKPT $token_num $token_complement