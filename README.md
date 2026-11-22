# Math Reasoning with GRPO: Qwen3-1.7B on GSM8K

Training Qwen3-1.7B to solve grade-school math problems with **Group Relative Policy Optimization (GRPO)**, a reinforcement learning method that needs only a reward function rather than labeled reasoning traces.

## Results

| Model | Token budget | Format rate | Accuracy |
|---|---|---|---|
| Qwen3-1.7B (base) | 512 | 14.6% | 30.6% |
| Qwen3-1.7B + GRPO (Run 1) | 512 | 71.0% | 68.8% |
| Qwen3-1.7B (base) | 1024 | 36.4% | 56.7% |
| **Qwen3-1.7B + GRPO (Run 2)** | **1024** | **84.5%** | **79.9%** |

Evaluated on the full GSM8K test set (1,319 problems), using greedy decoding. The token budget is the same for training (`max_completion_length`) and evaluation (`max_new_tokens`). Every model was evaluated with the same `evaluate.py` setup (Qwen3's default chat template with thinking mode on).

**Best result: 79.9% accuracy, +23.2 points over the base model at the same 1024-token budget, from 100 GRPO steps that saw only 400 training questions (about 5% of one epoch).**

**Trained model (Run 2):** [Huggyb/qwen3-1.7b-grpo-gsm8k](https://huggingface.co/Huggyb/qwen3-1.7b-grpo-gsm8k) on the Hugging Face Hub. To evaluate it directly:

```bash
python evaluate.py Huggyb/qwen3-1.7b-grpo-gsm8k
```

## Key findings

1. **GRPO adds +23.2 points** of accuracy over the base model when the token budget is held fixed at 1024.
2. **The token budget was the base model's biggest bottleneck.** Doubling it from 512 to 1024, with no training at all, raises base accuracy from 30.6% to 56.7%. Qwen3's thinking mode often ran past 512 tokens before writing an answer.
3. **At 512 tokens, RL taught the model to be concise.** Truncated completions dropped from ~55% to ~15% during training, without any explicit length penalty. This inflated Run 1's apparent gain (+38.2 points), since much of it came from learning to finish within the budget rather than from better reasoning.
4. **At 1024 tokens, the model learned faster but plateaued by step 50.** By step 100, every group of 8 completions got identical rewards (`frac_reward_zero_std` = 1.0), so the gradient was zero. GSM8K had stopped giving a learning signal.
5. **Truncation is still the main remaining failure.** About 95% of formatted answers are correct. Most errors are completions that never reach `<answer>`.

## Setup

| | |
|---|---|
| Model | `Qwen/Qwen3-1.7B` (full fine-tuning) |
| Dataset | `openai/gsm8k` (7,473 train / 1,319 test) |
| Library | TRL 1.18.2, Transformers 5.24.0, PyTorch 2.14.1 |
| Hardware | 1× A100-SXM4 40GB (Vast.ai) |
| Steps | 100 |
| Questions per step | 4 (× 8 generations = 32 completions) |
| Max completion length | 512 (Run 1) / 1024 (Run 2) |
| Learning rate | 1e-6, linear decay |
| Temperature | 1.0 |
| Precision | bf16, gradient checkpointing |
| In-training eval | 200 test questions, every 50 steps |

### Prompt format
The system prompt asks the model to reason inside `<think> </think>` and to put only the final number inside `<answer> </answer>`.

### Reward function (`reward.py`)
| Component | Reward |
|---|---|
| Output exactly matches `<think>…</think><answer>…</answer>` | +0.5 (otherwise −0.5) |
| Number in `<answer>` equals the ground truth | +1.0 (otherwise −1.0) |

Range: −1.5 to +1.5.

## Project structure
| File | Purpose |
|---|---|
| `prepare.py` | Loads GSM8K, extracts final answers, builds chat prompts, saves to `data/gsm8k_clean` |
| `reward.py` | Format and correctness reward functions |
| `train.py` | GRPO training with TRL |
| `evaluate.py` | Greedy-decoding evaluation on the GSM8K test set |
| `main.ipynb` | Exploration notebook used while building the pipeline |

## How to run

Requires a GPU with enough memory for full fine-tuning of a 1.7B model (an A100 40GB was used here).

```bash
pip install -r requirements.txt   # on a machine with CUDA already set up, leave torch out to keep the existing build
python -c "from prepare import save_data; save_data()"
python evaluate.py                          # baseline (base Qwen3-1.7B)
python train.py                             # GRPO training
python evaluate.py outputs/qwen3-grpo-v2/final
```

`train.py` and `evaluate.py` are configured for Run 2 (1024 tokens). To reproduce Run 1, set `max_completion_length=512` in `train.py` and `max_new_tokens=512` in `evaluate.py`.

## Experiments

| Run | Change | Token budget | Format | Accuracy |
|---|---|---|---|---|
| Baseline | n/a | 512 | 14.6% | 30.6% |
| Baseline | n/a | 1024 | 36.4% | 56.7% |
| Run 1 | GRPO, 100 steps | 512 | 71.0% | 68.8% |
| Run 2 | GRPO, 100 steps | 1024 | 84.5% | 79.9% |

Run 2 changes only the token budget relative to Run 1, so the comparison isolates the effect of the budget.

| Comparison | Accuracy change | What it isolates |
|---|---|---|
| Base 512 → Run 1 | 30.6% → 68.8% (+38.2) | GRPO at 512, including learning to fit the budget |
| Base 512 → Base 1024 | 30.6% → 56.7% (+26.1) | The budget alone, no training |
| Base 1024 → Run 2 | 56.7% → 79.9% (**+23.2**) | What GRPO adds when truncation is mostly removed |
| Run 1 → Run 2 | 68.8% → 79.9% (+11.1) | What the larger budget adds to a trained model |

**Format vs accuracy.** In the base model at 1024 tokens, accuracy (56.7%) is about 20 points above the format rate (36.4%). Those completions contain a correct `<answer>` but fail the strict `fullmatch` format check, so something appears outside the expected tags. After GRPO, the format rate (84.5%) is above accuracy (79.9%), so the format problem is largely solved: about 4.6% of answers are formatted but wrong, and about 15.5% are unformatted (mostly truncated).

## Analysis of Run 1 (512 tokens)

### 1. The model learned
- **Training reward** rose from about −0.8 to about +1.0 (smoothed). The fastest gains came between steps 20 and 50, followed by a plateau between +0.5 and +1.0.
- **Eval reward** on 200 held-out test questions rose from 0.34 (step 50) to 0.51 (step 100). It was still rising when training ended, with no sign of overfitting.

### 2. How it learned: finishing within the token budget

| Metric | Start | End |
|---|---|---|
| Truncated completions (`completions/clipped_ratio`) | ~55% | ~15% |
| Mean completion length | ~450 tokens | ~340 tokens |
| Min completion length | ~300 tokens | ~150–215 tokens |

The steepest drop in truncation (steps 25–50) lines up with the steepest rise in reward. **The model learned to reason more concisely without any explicit length penalty**, because truncated answers scored −1.5.

The mean length of *finished* completions fell from about 400 to about 300 tokens, then rose again to about 330 after step 80. Once the model reliably finished, it may have started spending slightly more of its budget on harder problems. This is a hypothesis, not a confirmed result.

### 3. Remaining failures were mostly truncation
About 32% of eval completions were still truncated at step 100, while `evaluate.py` showed 71.0% format and 68.8% accuracy: nearly every completion that finished was correct. (In-training eval samples at temperature 1, while `evaluate.py` is greedy, so the numbers are only roughly comparable.)

### 4. The learning signal ran low
- `frac_reward_zero_std` rose from about 0.4 to about 0.8: by the end, about 80% of question groups had all 8 completions scoring the same (mostly all correct). Those groups have zero advantage and contribute no gradient.
- `reward_std` peaked at about 1.4 around steps 20–35, when the model succeeded on some completions and failed on others, then declined.
- Together with the learning rate decaying to 0, this explains the plateau after step 50.

### 5. Training was stable
- `grad_norm` declined smoothly from about 2.5–3 to about 1.5, with no spikes.
- `entropy` declined gently from about 0.095 to about 0.077, with no sign of collapse.
- `loss` stayed between 0.01 and 0.05. In GRPO the loss is not a progress metric, because advantages are centered around zero within each group.
- `clip_ratio/*` was 0 throughout. This is expected: there is one optimizer step per generation batch (`num_iterations=1`), so the new/old policy ratio is always 1 and PPO-style clipping never triggers. (Not to be confused with `completions/clipped_ratio`, which measures truncated text.)

## Analysis of Run 2 vs Run 1

| Metric | Run 1 (512) | Run 2 (1024) |
|---|---|---|
| Eval reward, step 50 → 100 | 0.34 → 0.51 | 0.91 → 0.93 |
| Eval truncated completions, step 100 | 32% | 17% |
| Eval mean length of finished completions | ~295 tokens | ~476 tokens |
| Train `frac_reward_zero_std` (end, smoothed) | ~0.80 | ~0.88 (1.0 at step 100) |
| Eval `frac_reward_zero_std`, step 100 | 0.58 | 0.70 |
| Train entropy (end, smoothed) | 0.077 | 0.090 |
| Training step time | ~33 s | ~45 s |
| Total runtime | 3h 9m | ~4h 20m |

### 1. More room meant faster learning
Run 2's training reward crossed zero around step 10, while Run 1 took about 30 steps. At the start, `frac_reward_zero_std` was about 0.1–0.2 in Run 2 versus 0.3–0.5 in Run 1. With 1024 tokens, more early completions finished and got the right answer, so more groups had a mix of good and bad completions. That gives GRPO a non-zero advantage to learn from.

### 2. The model didn't need to compress its reasoning
At 512 tokens, the model shrank its finished answers to about 295 tokens to fit. With 1024, finished answers stayed around 476 tokens, about 60% longer reasoning. Run 1's forced compression probably cost some accuracy, which is consistent with Run 2 scoring 11 points higher.

### 3. Run 2 plateaued by step 50, because it ran out of learning signal
- Eval reward barely moved between step 50 (0.913) and step 100 (0.928), unlike Run 1, which was still rising.
- By the end, about 88% of training groups had identical rewards for all 8 completions (almost all correct). At step 100 it hit **1.0**: every group was uniform, so the advantage was zero, and `loss` and `grad_norm` were exactly 0 for that step.
- At 1024 tokens, the model has mostly learned what GSM8K at 8 generations per question can teach it. More steps would add little; further gains need harder problems or more generations per question.

### 4. Truncation is still the main remaining failure
About 17% of eval completions were still truncated at 1024 tokens, which closely matches the 15.5% of `evaluate.py` completions that weren't formatted. Of the completions that were formatted, about 95% were correct (79.9 / 84.5).

### 5. Entropy stayed higher
Entropy ended at about 0.090 in Run 2 versus 0.077 in Run 1. With more room, the model kept more variety in its outputs and showed no sign of collapse.

## Compute cost

| | Run 1 | Run 2 |
|---|---|---|
| Training steps | ~55 min (~33 s/step) | ~75 min (~45 s/step) |
| Two in-training eval passes | ~2h 9m (2 × ~3,870 s) | ~3h 5m (2 × ~5,540 s) |
| Total runtime | 3h 9m | ~4h 20m |
| Share spent on eval | ~68% | ~70% |

Doubling the budget made each step only 1.37× slower, because many completions finish well before 1024 tokens. In-training eval generates 8 completions for each of 200 prompts with plain Hugging Face `generate()` and dominated the runtime in both runs.

## Future work
- A larger token budget (1536–2048), since truncation is still the main failure at 1024
- Harder training data (e.g. the MATH dataset), since GSM8K stopped giving a learning signal by step 50 at 1024 tokens
- More generations per question (e.g. 16), to recover a learning signal from questions the model usually gets right
- Cheaper in-training eval (fewer prompts, larger eval batch) and vLLM generation to cut runtime
- Compare against Qwen3's `enable_thinking=False` mode
