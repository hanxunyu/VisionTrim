#!/bin/bash
run_sqa() {
    CUDA_VISIBLE_DEVICES=$GPU_ID bash -c "
        python -m llava.eval.model_vqa_science \
            --model-path $CKPT \
            --question-file /visiontrim/playground/data/eval/sqa/llava_test_CQM-A.json \
            --image-folder /visiontrim/playground/data/eval/sqa/test \
            --answers-file ../data/eval/scienceqa/answers/$method/$token_num.jsonl \
            --single-pred-prompt \
            --temperature 0 \
            --method $method \
            --DVTS_token_num $token_num \
            --TGVC_token_num $token_complement \
            --dataset-name sqa \
            --conv-mode vicuna_v1
        python llava/eval/eval_science_qa.py \
            --base-dir /visiontrim/playground/data/eval/sqa \
            --result-file ../data/eval/scienceqa/answers/$method/$token_num.jsonl \
            --output-file ../data/eval/scienceqa/answers/$method/$token_num-output.jsonl \
            --output-result ../data/eval/scienceqa/answers/$method/$token_num-result.json
    " #> "$ROOT_LOG/${LOG_PREFIX}.out" 2> "$ROOT_LOG/${LOG_PREFIX}.err" &
}

method=VisionTrim
CKPT=/llm_checkpoints/llava-v1.5-7b
GPU_ID=3
token_num=$1
token_complement=$2

run_sqa $GPU_ID $method $CKPT $token_num $token_complement