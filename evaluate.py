import sys
import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from reward import is_formatted, is_correct

model_path = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-1.7B"

test_ds = load_from_disk("data/gsm8k_clean")['test']

tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map="cuda")
model.eval()

batch_size = 32

n_format = 0
n_correct = 0

for start in range(0,len(test_ds), batch_size):
    batch = test_ds[start : start + batch_size]

    text = [
        tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
        for p in batch["prompt"]
    ]

    inputs = tokenizer(text, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=False)

    new_ids = output_ids[:, inputs["input_ids"].shape[1]:]
    completions = tokenizer.batch_decode(new_ids, skip_special_tokens=True)

    for out, gold in zip(completions, batch["answer"]):
        n_format += is_formatted(out)
        n_correct += is_correct(out, gold)

    print(f"{start + len(completions)}/{len(test_ds)} done")


print(f"format rate: {n_format / len(test_ds):.1%}")
print(f"accuracy:    {n_correct / len(test_ds):.1%}")