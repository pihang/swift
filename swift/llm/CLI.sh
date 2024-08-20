# CUDA_VISIBLE_DEVICES=0,1 \
# PYTHONPATH=/home/ph/LLM/swift-main/ \
# NPROC_PER_NODE=2 \
# python swift/cli/sft.py \
#     --model_type qwen-14b-chat \
#     --output_dir output \
#     --dataset ms-bench \

# --------自定义微调数据集
CUDA_VISIBLE_DEVICES=0,1 \
PYTHONPATH=/home/ph/LLM/swift-main/ \
NPROC_PER_NODE=2 \
python swift/cli/sft.py \
    --model_type qwen1half-14b-chat-int4 \
    --output_dir output \
    --custom_train_dataset_path /home/ph/LLM/swift-main/datasets/ms_bench/ms_agent_bench_v1_sft_sample.jsonl \
