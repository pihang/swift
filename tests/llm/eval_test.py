import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,7'
import sys
sys.path.append('/home/ph/LLM2/swift')
sys.path.insert(0,'/home/ph/LLM2/swift')
from swift.llm import eval_main,EvalArguments
Eval=EvalArguments(model_type='qwen1half-7b-chat',eval_dataset=['arc'],eval_limit=20,infer_backend='pt')  

eval_main(Eval)