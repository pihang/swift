# Experimental environment: A100
# PYTHONPATH=../../.. \
CUDA_VISIBLE_DEVICES=4,5,6,7 \
swift infer \
    --ckpt_dir "output/baichuan2-13b-chat/lora-mp-ddp-ds/baichuan2-13b-chat/v0-20240507-171254/checkpoint-1107" \
    --load_dataset_config true \
    --max_new_tokens 2048 \
    --temperature 0.1 \
    --top_p 0.7 \
    --repetition_penalty 1. \
    --do_sample true \
    --merge_lora false \
