# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'
# import sys
# sys.path.append('/home/ph/LLM2/swift')
# sys.path.insert(0,'/home/ph/LLM2/swift')

# from swift.llm import (
#     get_model_tokenizer, get_template, inference, ModelType, get_default_template_type,
# )
# from swift.utils import seed_everything

# model_type = ModelType.qwen1half_14b_chat_int4   # 
# template_type = get_default_template_type(model_type)
# print(f'template_type: {template_type}')  # template_type: qwen

# kwargs = {}
# # kwargs['use_flash_attn'] = True  # 使用flash_attn

# model, tokenizer = get_model_tokenizer(model_type, model_kwargs={'device_map': 'auto'}, **kwargs)
# # 修改max_new_tokens
# model.generation_config.max_new_tokens = 128

# template = get_template(template_type, tokenizer)
# seed_everything(42)
# query = '浙江的省会在哪里？'
# response, history = inference(model, template, query)
# print(f'query: {query}')
# print(f'response: {response}')
# query = '这有什么好吃的？'
# response, history = inference(model, template, query, history, verbose=True, stream=True)
# print(f'\nquery: {query}')
# print(f'response: {response}')
# print(f'history: {history}')
import sys
sys.path.append('/home/ph/LLM2/swift')
sys.path.insert(0,'/home/ph/LLM2/swift')
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'

from swift.llm import DatasetName, ModelType, SftArguments, sft_main

sft_args = SftArguments(
    model_type=ModelType.qwen_14b_chat,
    # model_id_or_path='/home/ph/LLM/Qwen1.5/Qwen1.5-14B-Chat-GPTQ-Int4',  # model_cache_dir='/home/ph/LLM/Qwen1.5/Qwen1.5-14B-Chat-GPTQ-Int4',
    dataset=[DatasetName.alpaca_zh,DatasetName.alpaca_en],    # ,firefly_zh
    train_dataset_sample=1000,
    logging_steps=5,
    max_length=2048,
    learning_rate=5e-5,
    warmup_ratio=0.4,
    output_dir='output',
    lora_target_modules=['ALL'],   # 在所有的linear层(包括qkvo以及mlp)加lora. 这通常是效果最好的.
    self_cognition_sample=500,
    model_name=['猴乐乐', 'houlele'],
    model_author=['同元智算', 'Tongyuan intelligence calculation'])
output = sft_main(sft_args)
best_model_checkpoint = output['best_model_checkpoint']
print(f'best_model_checkpoint: {best_model_checkpoint}')