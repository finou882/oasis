import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("stockmark/stockmark-100b")
model = AutoModelForCausalLM.from_pretrained("stockmark/stockmark-100b", device_map="auto", torch_dtype=torch.bfloat16)

input_ids = tokenizer("What is Geneleting AI", return_tensors="pt").input_ids.to(model.device)
with torch.inference_mode():
    tokens = model.generate(
        input_ids,
        max_new_tokens = 256,
        do_sample = True,
        temperature = 0.7,
        top_p = 0.95,
        repetition_penalty = 1.08
    )
    
output = tokenizer.decode(tokens[0], skip_special_tokens=True)