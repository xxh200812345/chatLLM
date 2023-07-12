import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

open_calm_model="cyberagent/open-calm-1b"

# 导入所需库和模型
model = AutoModelForCausalLM.from_pretrained(open_calm_model, device_map="auto", torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(open_calm_model)

# 准备输入
inputs = tokenizer("AIによって私達の暮らしは、", return_tensors="pt").to(model.device)

# 生成文本
with torch.no_grad():
    tokens = model.generate(
        **inputs,                       # 输入参数
        max_new_tokens=128,             # 生成的最大令牌数
        do_sample=True,                 # 是否使用采样方式生成文本
        temperature=0.7,                # 采样温度，控制生成文本的多样性
        top_p=0.9,                      # 采样时选择概率最高的token的阈值
        repetition_penalty=1.05,        # 重复惩罚，控制生成文本中重复内容的程度
        pad_token_id=tokenizer.pad_token_id,   # 用于填充的token的id
    )

# 解码生成的标记序列为文本
output = tokenizer.decode(tokens[0], skip_special_tokens=True)

# 打印输出
print(output)
