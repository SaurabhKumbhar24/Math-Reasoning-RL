from datasets import load_dataset

def extract_thinking(data):
    answer_parts = data['answer'].split("####")
    answer = answer_parts[1].strip().replace(",","")
    thinking = answer_parts[0].strip()
    return {'question': data['question'],'thinking':thinking,'answer':answer}

system_prompt = (
    "Solve the math problem below.\n"
    "First, reason step by step inside <think> </think> tags.\n"
    "Then give the final answer inside <answer> </answer> tags.\n"
    "The answer must contain only the final number, with no units, symbols, or commas.\n"
    "Format:\n"
    "<think>\n...\n</think>\n"
    "<answer>\n...\n</answer>"
)
def create_prompt(data):
    return {"prompt":[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": data['question']}, 
    ]}

def save_data():
    ds = load_dataset("openai/gsm8k", "main")
    ds = ds.map(extract_thinking)
    clean_ds = ds.map(create_prompt)
    clean_ds.save_to_disk("data/gsm8k_clean")