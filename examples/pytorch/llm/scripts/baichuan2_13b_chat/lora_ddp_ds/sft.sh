# Experimental environment: 2 * A100
# 2 * 35GB GPU memory
# 本来dataset = damo-agent-mini-zh
nproc_per_node=4

PYTHONPATH='/home/ph/LLM2/swift/' \
CUDA_VISIBLE_DEVICES=1,2,3,4 \
torchrun \
    --nproc_per_node=$nproc_per_node \
    --master_port 29500 \
    swift/cli/sft.py \
    --model_type 'baichuan2-13b-chat' \
    --model_id_or_path '/home/ph/LLM/Baichuan-13B-main/Baichuan2-13B-Chat' \
    --model_revision master \
    --sft_type lora \
    --tuner_backend peft \
    --template_type AUTO \
    --dtype AUTO \
    --output_dir output/baichuan2-13b-chat/lora-mp-ddp-ds \
    --ddp_backend nccl \
    --dataset dureader-robust-zh \
    --num_train_epochs 1 \
    --max_length 4096 \
    --check_dataset_strategy warning \
    --lora_rank 8 \
    --lora_alpha 32 \
    --lora_dropout_p 0.05 \
    --lora_target_modules ALL \
    --gradient_checkpointing true \
    --batch_size 1 \
    --weight_decay 0.1 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps $(expr 16 / $nproc_per_node) \
    --max_grad_norm 0.5 \
    --warmup_ratio 0.03 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 10 \
    --deepspeed default-zero3 \
