CUDA_VISIBLE_DEVICES=0 \
python src/llm_infer.py \
    --model_type qwen-7b-chat \
    --sft_type lora \
    --template_type chatml \
    --dtype bf16 \
    --ckpt_dir "output/qwen-7b-chat/vx_xxx/checkpoint-xxx" \
    --eval_human false \
    --dataset cot-en,cot-zh \
    --max_length 2048 \
    --use_flash_attn true \
    --max_new_tokens 1024 \
    --temperature 0.9 \
    --top_k 20 \
    --top_p 0.9 \
    --do_sample true \