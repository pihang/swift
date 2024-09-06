# Merge LoRA增量权重并推理
# 如果你需要量化, 可以指定`--quant_bits 4`.
CUDA_VISIBLE_DEVICES=4,5,6,7 swift export \
    --ckpt_dir 'output/baichuan2-13b-chat/lora-mp-ddp-ds/baichuan2-13b-chat/v0-20240507-171254/checkpoint-1107' --merge_lora true