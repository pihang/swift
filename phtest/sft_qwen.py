import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'
import sys
sys.path.append('/home/ph/LLM/swift-main')
sys.path.insert(0,'/home/ph/LLM/swift-main')
from swift.llm import DatasetName, ModelType, SftArguments, sft_main

sft_args = SftArguments(
    model_type=ModelType.qwen_14b_chat,
    # model_id_or_path='/home/ph/LLM/Qwen1.5/Qwen1.5-14B-Chat-GPTQ-Int4',  model_cache_dir='/home/ph/LLM/Qwen1.5/Qwen1.5-14B-Chat-GPTQ-Int4',
    dataset=[DatasetName.alpaca_zh],
    train_dataset_sample=1000,
    logging_steps=5,
    max_length=2048,
    warmup_ratio=0.4,
    output_dir='output',
    lora_target_modules=['ALL'],   # 在所有的linear层(包括qkvo以及mlp)加lora. 这通常是效果最好的.
    self_cognition_sample=500,
    model_name=['猴乐乐', 'houlele'],
    model_author=['同元智算', 'Tongyuan intelligence calculation'])
output = sft_main(sft_args)
best_model_checkpoint = output['best_model_checkpoint']
print(f'best_model_checkpoint: {best_model_checkpoint}')