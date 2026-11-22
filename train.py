import os
from trl import GRPOConfig, GRPOTrainer
from datasets import load_from_disk
from prepare import save_data
from reward import batch_reward

if (not os.path.exists("data/gsm8k_clean")):
    save_data()

ds = load_from_disk("data/gsm8k_clean")

training_args = GRPOConfig(
    #Training Hyperparameters
    output_dir="outputs/qwen3-grpo",
    num_generations=8,
    temperature=1.,
    max_completion_length=1024,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    learning_rate=1e-6,
    max_steps=100,

    #Optimization and Memory Management
    bf16=True,
    gradient_checkpointing=True,

    #Logging
    logging_steps=1,
    report_to="tensorboard",
    run_name="qwen3-1.7b-gsm8k-try2",
    log_completions=True, 

    #Saving
    save_strategy="steps",
    save_steps=50,
    save_total_limit=2,

    #Validation
    eval_strategy="steps",
    eval_steps=50,
    per_device_eval_batch_size=8,
    
)

trainer = GRPOTrainer(
    model="Qwen/Qwen3-1.7B",
    reward_funcs=[batch_reward],
    args=training_args,
    train_dataset=ds["train"],
    eval_dataset=ds['test'].select(range(200))
)

trainer.train() #resume_from_checkpoint=True on second run
trainer.save_model("outputs/qwen3-grpo-v2/final")