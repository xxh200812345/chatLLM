from transformers import AutoTokenizer, AutoModel
import torch

chatglm_6b_path = r'H:\vswork\Machinelearning\chatglm-6b\res\chatglm-6b-int4'

# 清空GPU缓存
torch.cuda.empty_cache()

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(chatglm_6b_path, trust_remote_code=True, revision="")

# 加载模型，并将模型转换为半精度浮点数，并移动到GPU上
model = AutoModel.from_pretrained(chatglm_6b_path, trust_remote_code=True, revision="").half().cuda()

# 设置模型为评估模式
model = model.eval()

# 进行对话生成
response, history = model.chat(tokenizer, "介绍一下你自己", history=[])
print(response)

# 继续对话生成
response, history = model.chat(tokenizer, "请问你能再重复一遍吗？谢谢！", history=history)
print(response)
