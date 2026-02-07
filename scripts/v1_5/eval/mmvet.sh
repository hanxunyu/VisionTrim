#!/bin/bash
run_mmvet() {
    CUDA_VISIBLE_DEVICES=$GPU_ID bash -c "
        python -m llava.eval.model_vqa \
            --model-path $CKPT \
            --question-file /playground/data/mm-vet/llava-mm-vet.jsonl \
            --image-folder /playground/data/mm-vet/images \
            --answers-file ../data/eval/mm-vet/answers/$method/$token_num.jsonl \
            --temperature 0 \
            --method $method \
            --dataset-name mmvet \
            --DVTS_token_num $token_num \
            --TGVC_token_num $token_complement \
            --conv-mode vicuna_v1

        mkdir -p ../data/eval/mm-vet/results/$method
        python scripts/convert_mmvet_for_eval.py \
            --src ../data/eval/mm-vet/answers/$method/$token_num.jsonl \
            --dst ../data/eval/mm-vet/results/$method/$token_num.json
    " #> "$ROOT_LOG/${LOG_PREFIX}.out" 2> "$ROOT_LOG/${LOG_PREFIX}.err" &
}

method=VisionTrim
CKPT=/LLM_checkpoint/llava-v1.5-7b
GPU_ID=3
token_num=$1
token_complement=$2

run_mmvet $GPU_ID $method $CKPT $token_num $token_complement