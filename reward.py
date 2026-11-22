import re

def extract_number_and_unit(text):
    pattern = r"([$€£¥]?)\s*([-+]?\d[\d,]*\.?\d*)\s*([a-zA-Z°%/]*)"
    match = re.search(pattern, text)
    
    if not match:
        return [None, None]
        
    prefix_unit, number_str, suffix_unit = match.groups()
    number = float(number_str.replace(",", ""))
    unit = prefix_unit or suffix_unit
    return [number, unit]

def is_formatted(gen_answer):
    if re.fullmatch(r"\s*<think>.*?</think>\s*<answer>.*?</answer>\s*", gen_answer, re.DOTALL):
        return True
    return False

def is_correct(gen_answer, actual_answer):
    match = re.search(r"<answer>(.*?)</answer>", gen_answer, re.DOTALL)
    ans = extract_number_and_unit(match.group(1).strip())[0] if match else None
    if(ans is not None and abs(float(ans) - float(actual_answer)) < 1e-6):
        return True
    return False

#Reward Function
def _reward(gen_answer, actual_answer):
    reward = 0.
    #Format Reward
    if is_formatted(gen_answer):
        reward += .5
    else:
        reward -= .5

    #Compare Answers Reward
    if is_correct(gen_answer, actual_answer):
        reward += 1
    else:
        reward -= 1

    return reward

def batch_reward(completions, answer, **kwargs):
    reward_b = []
    for i in range(len(completions)):
        reward_b.append(_reward(completions[i][0]["content"], answer[i]))
    return reward_b