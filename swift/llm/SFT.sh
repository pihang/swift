# CUDA_VISIBLE_DEVICES=0,1 \
# PYTHONPATH='/home/ph/LLM2/swift' \
# NPROC_PER_NODE=2 \
# python swift/cli/sft.py\
#     --model_type qwen-14b-chat \
#     --dataset ms-bench alpaca-en \
#     --train_dataset_sample 1000 \
#     --logging_steps 5 \
#     --max_length 2048 \
#     --learning_rate 5e-5 \
#     --warmup_ratio 0.4 \
#     --output_dir output \
#     --lora_target_modules ALL \
#     --self_cognition_sample 500 \
#     --model_name 小黄 'Xiao Huang' \
#     --model_author 魔搭 ModelScope \

# --------自定义微调数据集
CUDA_VISIBLE_DEVICES=2,3 \
PYTHONPATH='/home/ph/LLM2/swift/' \
NPROC_PER_NODE=2 \
python swift/cli/sft.py \
    --model_type qwen1half-14b-chat-int4 \
    --output_dir output \
    --custom_train_dataset_path /home/ph/LLM/swift-main/datasets/ms_bench/ms_agent_bench_v1_sft_sample.jsonl 