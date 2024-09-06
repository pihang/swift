# 使用arc评测，每个子数据集限制评测10条，推理backend使用pt
swift eval \
    --model_type "qwen1half-7b-chat" \
    --model_id_or_path "/home/ph/LLM/Qwen1.5/Qwen1.5-7B-Chat" \
    --eval_dataset arc \
    --eval_limit 10 \
    --infer_backend pt   