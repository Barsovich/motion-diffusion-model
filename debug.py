from transformers import AutoTokenizer, GPT2Model
import torch

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

model = GPT2Model.from_pretrained("gpt2")

for param in model.parameters():
    param.requires_grad = False

inputs = tokenizer(["Hello, my dog is cute.", "Hello, is cute."], padding=True, return_tensors="pt")
print(inputs)
outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state
print(last_hidden_states.shape)


# # Text
# [bs, 18, 768]
# [bs, 18, 64]
## dilate
# [bs, 60, 64]
## group
# [bs, 10, 384]
# [bs, 10, 30]

# # Audio
# [bs, 150, 30]
## group
# [bs, 10, 30 * 15]
# [bs, 10, 70]


# [bs, 10, 70]
# [bs, 10, 30]
# [bs, 10, 100]
# [bs, 1000]


