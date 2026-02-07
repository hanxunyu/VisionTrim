#!/bin/bash

# CKPT="llava-v1.5-7b"
CKPT="llava-v1.5-7b"
method=VisionTrim
token_num=$1
token_complement=$2
GPU_ID=0

CUDA_VISIBLE_DEVICES=$GPU_ID python -W ignore -m llava.eval.model_vqa_loader \
    --model-path /llm_checkpoints/${CKPT} \
    --question-file /visiontrim/playground/data/eval/MME/llava_mme.jsonl \
    --image-folder /visiontrim/playground/data/eval/MME/MME_Benchmark_release_version/MME_Benchmark \
    --answers-file /visiontrim/playground/data/eval/MME/answers/$method/$token_num.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --method $method \
    --dataset-name mme \
    --DVTS_token_num $token_num \
    --TGVC_token_num $token_complement

cd /visiontrim/playground/data/eval/MME

python convert_answer_to_mme.py --experiment ${method}/$token_num+$method

cd eval_tool

python calculation.py --results_dir answers/${method}/$token_num+$method