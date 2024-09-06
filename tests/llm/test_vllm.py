import os
import sys
sys.path.append('/home/ph/LLM2/swift')
sys.path.insert(0,'/home/ph/LLM2/swift')
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from swift.llm import AppUIArguments, ModelType, app_ui_main

app_ui_args = AppUIArguments(model_type=ModelType.qwen_14b_chat)   # quantization_bit=4
app_ui_main(app_ui_args)