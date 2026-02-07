#!/bin/bash
run_pope() {
    CUDA_VISIBLE_DEVICES=$GPU_ID bash -c "
        python -m llava.eval.model_vqa_loader \
            --model-path $CKPT \
            --question-file /visiontrim/playground/data/eval/pope/llava_pope_test.jsonl \
            --image-folder /visiontrim/playground/data/eval/pope/val2014 \
            --answers-file ../data/eval/pope/answers/$method/$token_num.jsonl \
            --temperature 0 \
            --conv-mode vicuna_v1 \
            --dataset-name pope \
            --method $method \
            --DVTS_token_num $token_num \
            --TGVC_token_num $token_complement

            
        python llava/eval/eval_pope.py \
            --annotation-dir /visiontrim/playground/data/eval/pope/coco \
            --question-file /visiontrim/playground/data/eval/pope/llava_pope_test.jsonl \
            --result-file ../data/eval/pope/answers/$method/$token_num.jsonl
    " #> "$ROOT_LOG/${LOG_PREFIX}.out" 2> "$ROOT_LOG/${LOG_PREFIX}.err" &
}

method=VisionTrim
CKPT=/llm_checkpoints/llava-v1.5-7b
GPU_ID=1
token_num=$1
token_complement=$2

run_pope $GPU_ID $method $CKPT $token_num $token_complement