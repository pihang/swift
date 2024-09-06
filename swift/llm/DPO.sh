nproc_per_node=2
NCCL_IB_DISABLE=1 \
NCCL_P2P_DISABLE=1 \
CUDA_VISIBLE_DEVICES=1,2 \
PYTHONPATH=/home/ph/LLM/swift-main/ \
NPROC_PER_NODE=$nproc_per_node \
MASTER_PORT=29500 \
python swift/llm/rlhf.py \
    --rlhf_type dpo \
    --model_type  qwen2-0_5b-instruct \
    --ref_model_type  qwen2-0_5b-instruct \
    --model_revision  master \
    --sft_type  lora \
    --tuner_backend  swift \
    --dtype  AUTO  \
    --output_dir  output  \
    --dataset  hh-rlhf-cn:harmless_base_cn  \
    --num_train_epochs  3  \
    --max_length  1024  \
    --max_prompt_length  512  \
    --check_dataset_strategy  none  \
    --lora_rank  8  \
    --lora_alpha  32  \
    --lora_dropout  0.05  \
    --lora_target_modules  ALL  \
    --gradient_checkpointing  true  \
    --batch_size  1  \
    --weight_decay  0.1  \
    --learning_rate  5e-5  \
    --gradient_accumulation_steps  $(expr 16 / $nproc_per_node)  \
    --max_grad_norm  1.0  \
    --warmup_ratio  0.03  \
    --eval_steps  2000  \
    --save_steps  2000  \
    --save_total_limit  2  \
    --logging_steps  10 \