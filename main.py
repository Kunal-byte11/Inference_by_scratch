from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os

os.makedirs("../weights", exist_ok=True)

model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

for name, param in model.named_parameters():
    print(name, tuple(param.shape)) 

    values = param.detach().cpu().numpy().flatten()

    with open(f"../weights/{name}.txt", "w") as f:
        f.write(" ".join(map(str, values)))

tokenizer.save_pretrained("../weights/tokenizer")