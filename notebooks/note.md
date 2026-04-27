

## Setup Environment
```bash
uv sync --extra=cu128
source .venv/bin/activate
```

Required: `hf auth login` with access to cosmos models

Optional: `export HF_HOME="/scratch/ows/.cache/huggingface"` to another place you like
 

## PAI Benchmark

### Run PAI Benchmark

Generate and profile the inference
```bash
torchrun --nproc_per_node=8 examples/inference_pai.py -i 0 1 2 -o outputs/pai_eval_output/test --model 2B/post-trained --seed 42 --profile
```
<!-- 
Run on specific GPUs
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 examples/inference_pai.py -i 0 1 2 -o outputs/pai_eval_output/test --model 2B/post-trained --seed 42 --profile
```
Run a second batch in parallel
```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=29501 examples/inference_pai.py -i 3 4 5 -o outputs/pai_eval_output/test --model 2B/post-trained --seed 42 --profile
``` -->



