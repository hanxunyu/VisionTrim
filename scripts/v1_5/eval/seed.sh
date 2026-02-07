#!/bin/bash
export CUDA_VISIBLE_DEVICES=5,6
run_seed(){
    python -m llava.eval.model_vqa_loader \
            --model-path $CKPT \
            --question-file ../data/eval/seed_bench/llava-seed-bench.jsonl \
            --image-folder ../data/seed_bench \
            --answers-file ../data/eval/seed_bench/answers/$method/$token_num.jsonl \
            --num-chunks 1 \
            --chunk-idx 0 \
            --temperature 0 \
            --conv-mode vicuna_v1 \
            --DVTS_token_num $token_num \
            --TGVC_token_num $token_complement \
            --dataset-name seed \
            --method $method 
    wait

    output_file=../data/eval/seed_bench/answers/$method/$token_num.jsonl

    # # Clear out the output file if it exists.
    # > "$output_file"

    # # Loop through the indices and concatenate each file.
    # for IDX in $(seq 0 $((CHUNKS-1))); do
    #     cat  ../data/eval/seed_bench/answers/$NAME/${CHUNKS}_${IDX}.jsonl >> "$output_file"
    # done

    # Evaluate
    python scripts/convert_seed_for_submission.py \
        --annotation-file ../data/seed_bench/SEED-Bench.json \
        --result-file $output_file \
        --result-upload-file ../data/eval/seed_bench/answers_upload/$method/$token_num.jsonl
}

method=VisionTrim
token_num=$1
token_complement=$2
CKPT=/llm_checkpoints/llava-v1.5-7b

run_seed $GPU_ID $method $CKPT $token_num $token_complement 

