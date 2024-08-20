# Experimental environment: 3090
PYTHONPATH=/home/ph/LLM/swift-main \
CUDA_VISIBLE_DEVICES=0 \
python examples/pytorch/llm/llm_infer.py \
    --ckpt_dir "output/baichuan2-7b/vx_xxx/checkpoint-xxx" \
    --load_dataset_config true \
    --max_length 2048 \
    --max_new_tokens 2048 \
    --temperature 0.7 \
    --top_p 0.7 \
    --repetition_penalty 1. \
    --do_sample true \
    --merge_lora_and_save false \
