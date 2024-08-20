# from modelscope import snapshot_download
# model_dir = snapshot_download('ZhipuAI/glm-4-9b-chat-int4', cache_dir='./')


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cpu"

tokenizer = AutoTokenizer.from_pretrained("./ZhipuAI/glm-4-9b-chat", trust_remote_code=True)

query = "你好"

inputs = tokenizer.apply_chat_template([{"role": "user", "content": query}],
                                       add_generation_prompt=True,
                                       tokenize=True,
                                       return_tensors="pt",
                                       return_dict=True
                                       )

inputs = inputs.to(device)
model = AutoModelForCausalLM.from_pretrained(
    "./glm-4-9b-chat",
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    load_in_4bit=True
).eval()
model.save_pretrained("glm-4-9b-chat-int4")
tokenizer.save_pretrained("glm-4-9b-chat-int4")