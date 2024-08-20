import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
import sys
sys.path.append('/home/ph/LLM/swift-main')

from swift.llm import AppUIArguments, merge_lora, app_ui_main,ModelType, InferArguments, infer_main

# 先merge lora参数再加载
best_model_checkpoint = '/home/ph/LLM/swift-main/output/qwen-14b-chat/v0-20240222-151543/checkpoint-68'
app_ui_args = AppUIArguments(model_id_or_path='/home/ph/LLM/Qwen-14B-main/Qwen-14B-Chat',ckpt_dir=best_model_checkpoint)
merge_lora(app_ui_args, device_map='auto')
result = app_ui_main(app_ui_args)
