from transformers import pipeline
import torch

device = -1 # cpu
if torch.cuda.is_available():
    device = 0

# device 0 = cuda:0
generator = pipeline("text-generation", model="abeja/gpt-neox-japanese-2.7b", torch_dtype=torch.float16, device=device)
generated = generator(
    "優秀な若人さまが、人類史上最速で GPT をお極めなされるには、",
    max_length=300,
    do_sample=True,
    num_return_sequences=3,
    top_p=0.95,
    top_k=50
)
print(*generated, sep="\n")
