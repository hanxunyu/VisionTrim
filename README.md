<h2 align="center">
  <b>⚡️ VisionTrim: Unified Vision Token Compression for <br> Training-Free MLLM Acceleration</b>
  
  <b>[ICLR 2026]</b>
</h2>

This is an official repository for the paper "VisionTrim: Unified Vision Token Compression for Training-Free MLLM Acceleration".

<div align="left">
<img src="assets/pipeline-visiontrim.png" width="99%" alt="VisionTrim">
</div>

With two effective plug-and-play modules (DVTS and TGVC) that accelerate both vision encoding and LLM decoding stages, VisionTrim achieves **98.8%** of the original performance with an **88.9%** reduction ratio in token count **without additional training costs**, consistently surpassing previous SOTA methods across various reduction ratios in both image and video understanding tasks.
#
### 📰 News
* **`Jan. 26th, 2026`:** VisionTrim is accepted by ICLR 2026!

## ⚙️ Setup

### 🏝️ Environment

1. Clone this repository.

```bash
git clone https://github.com/hanxunyu/VisionTrim.git
cd VisionTrim
```

2. Install necessary packages.

```bash
conda create -n visiontrim python=3.10 -y
conda activate visiontrim
pip install -e .
```

3. (Optional) Install FlashAttention for further inference acceleration.

```bash
pip install flash-attn --no-build-isolation
```

### 📦️ Model

Download corresponding [LLaVA](https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md) checkpoints from [Hugging Face](https://huggingface.co/liuhaotian) 🤗:

| Version                |    LLM     |                          Checkpoint                          |
| ---------------------- | :--------: | :----------------------------------------------------------: |
| LLaVA-1.5              | Vicuna-7B  | [liuhaotian/llava-v1.5-7b](https://huggingface.co/liuhaotian/llava-v1.5-7b) |
| LLaVA-1.5              | Vicuna-13B | [liuhaotian/llava-v1.5-13b](https://huggingface.co/liuhaotian/llava-v1.5-13b) |
| LLaVA-1.6 (LLaVA-NeXT) | Vicuna-7B  | [liuhaotian/llava-v1.6-vicuna-7b](https://huggingface.co/liuhaotian/llava-v1.6-vicuna-7b) |
| LLaVA-1.6 (LLaVA-NeXT) | Vicuna-13B | [liuhaotian/llava-v1.6-vicuna-13b](https://huggingface.co/liuhaotian/llava-v1.6-vicuna-13b) |

### 📊 Data

Download each dataset according to [EVAL.md](EVAL.md).

## 🔬 Analysis

To analyze the inaccurate text-visual attention in VLMs, you need to download the visual instruction tuning data for [LLaVA](https://github.com/haotian-liu/LLaVA/tree/main?tab=readme-ov-file#visual-instruction-tuning) first, which we use for attention computation. And we provide the 1K subset for attention analysis in `./playground/data/analysis/llava_v1_5_mix1k.jsonl`.

### 🛹 Attention Shift

To analyze the attention shift in VLMs, run the script `./scripts/analyze_attn_shift.sh`.

```bash
bash scripts/v1_5/analyze_attn_shift.sh
```

### 🪩 Attention Dispersion

To analyze the attention dispersion in VLMs, run the script `./scripts/analyze_attn_dispersion.sh`.

```bash
bash scripts/v1_5/analyze_attn_dispersion.sh
```

## 📋️ Evaluation

The main implementation of our VisionTrim is mainly in [`llava_llama.py`](llava/model/language_model/llava_llama.py), [`clip_encoder.py`](llava/model/multimodal_encoder/clip_encoder.py), [`llava_arch.py`](llava/model/llava_arch.py), [`model_vqa.py`](llava/eval/model_vqa.py), and [`model_vqa_loader.py`](llava/eval/model_vqa_loader.py)

We provide the evaluation scripts for each benchmark under `./scripts/v1_5/eval`, you need to set the **start layer** and **remaining visual token number** as the bash argument. The detailed guidance for evaluation commands and online submission of each benchmark can be found in [EVAL.md](EVAL.md).

For evaluation with the 13B LLM, you just need to replace the `CKPT` argument from `llava-v1.5-7b` to `llava-v1.5-13b` in each script. And for evaluation with LLaVA-NeXT, you can use the scripts in `./scripts/v1_6/eval`. 

### GQA

1. Download the [data](https://cs.stanford.edu/people/dorarad/gqa/download.html) and [evaluation scripts](https://cs.stanford.edu/people/dorarad/gqa/evaluate.html) following the official instructions and put under `../data/gqa/data`. You may need to modify `eval.py` due to the missing assets in the GQA v1.2 release.
2. Single-GPU or Multi-GPU inference and evaluation.

```Shell
method=VisionTrim
bash scripts/v1_5/eval/gqa.sh $layer $token_num
```

### ScienceQA

1. Under `../data/scienceqa`, download `images`, `pid_splits.json`, `problems.json` from the `data/scienceqa` folder of the ScienceQA.
2. Single-GPU or Multi-GPU inference and evaluation.

```Shell
method=VisionTrim
bash scripts/v1_5/eval/sqa.sh $layer $token_num
```

### TextVQA

1. Download [`TextVQA_0.5.1_val.json`](https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_val.json) and [images](https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip) and extract to `../data/textvqa`.
2. Single-GPU or Multi-GPU inference and evaluation.

```Shell
method=VisionTrim
bash scripts/v1_5/eval/textvqa.sh $layer $token_num
```

### POPE

1. Download `coco` from [POPE](https://github.com/AoiDragon/POPE/tree/e3e39262c85a6a83f26cf5094022a782cb0df58d/output/coco) and put under `../data`.
2. Single-GPU or Multi-GPU inference and evaluation.

```Shell
method=VisionTrim
bash scripts/v1_5/eval/pope.sh $layer $token_num
```

### MMBench

1. Download [`mmbench_dev_20230712.tsv`](https://download.openmmlab.com/mmclassification/datasets/mmbench/mmbench_dev_20230712.tsv) and put under `../data/mmbench`.
2. Single-GPU or Multi-GPU inference and evaluation.

```Shell
method=VisionTrim
bash scripts/v1_5/eval/mmbench.sh $layer $token_num
```

3. Submit the results to the [evaluation server](https://opencompass.org.cn/leaderboard-multimodal): `../data/eval/mmbench/answers_upload/mmbench_dev_20230712`.

### MMBench-CN

1. Download [`mmbench_dev_cn_20231003.tsv`](https://download.openmmlab.com/mmclassification/datasets/mmbench/mmbench_dev_cn_20231003.tsv) and put under `../data/mmbench`.
2. Single-GPU or Multi-GPU inference and evaluation.

```Shell
method=VisionTrim
bash scripts/v1_5/eval/mmbench_cn.sh $layer $token_num
```

3. Submit the results to the evaluation server: `../data/eval/mmbench/answers_upload/mmbench_dev_cn_20231003`.


### SEED-Bench

1. Following the official [instructions](https://github.com/AILab-CVC/SEED-Bench/blob/main/DATASET.md) to download the images and the videos. Put images under `../data/seed_bench/SEED-Bench-image`. Note that we only use the image subset to evaluate.
2. Single-GPU or Multi-GPU inference and evaluation.

```Shell
method=VisionTrim
bash scripts/v1_5/eval/seed.sh $layer $token_num
```

### MM-Vet

1. Extract [`mm-vet.zip`](https://github.com/yuweihao/MM-Vet/releases/download/v1/mm-vet.zip) to `../data/mmvet`.
2. Single-GPU or Multi-GPU inference and evaluation.

```Shell
method=VisionTrim
bash scripts/v1_5/eval/mmvet.sh $layer $token_num
```

3. Evaluate the predictions in `../data/eval/mmvet/results` using the official Jupyter notebook.

## 😊 Acknowledgement

We are grateful for the open-source contributions of other projects:

- [LLaVA](https://github.com/haotian-liu/LLaVA)
- [Vicuna](https://github.com/lm-sys/FastChat)
- [Video-LLaVA](https://github.com/PKU-YuanGroup/Video-LLaVA)
- [FastV](https://github.com/pkunlp-icler/FastV)

## 🖊️ Citation

If you find our VisionTrim useful for your research, please consider giving this repository a star and citing our paper as follows:
```bibtex
@misc{
}
```
