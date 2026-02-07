#!/bin/bash

run_textvqa() {
    CUDA_VISIBLE_DEVICES=$GPU_ID bash -c "
        python -m llava.eval.model_vqa_loader \
            --model-path $CKPT \
            --question-file /visiontrim/playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
            --image-folder /visiontrim/playground/data/eval/textvqa/train_images \
            --answers-file ../data/eval/textvqa/answers/$method/$token_num.jsonl \
            --temperature 0 \
            --method $method \
            --DVTS_token_num $token_num \
            --TGVC_token_num $token_complement \
            --dataset-name textvqa \
            --conv-mode vicuna_v1

        python -m llava.eval.eval_textvqa \
            --annotation-file /visiontrim/playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
            --result-file ../data/eval/textvqa/answers/$method/$token_num.jsonl
    " #> "$ROOT_LOG/${LOG_PREFIX}.out" 2> "$ROOT_LOG/${LOG_PREFIX}.err" &
}

method=VisionTrim
CKPT=/llm_checkpoints/llava-v1.5-7b
GPU_ID=0
token_num=$1
token_complement=$2

run_textvqa $GPU_ID $method $CKPT $token_num $token_complement

