import yaml
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding

# Load config
with open("configs/config.yaml") as f:
    config = yaml.safe_load(f)

# Load dataset & tokenizer
dataset = load_dataset(config["dataset_name"])
tokenizer = AutoTokenizer.from_pretrained(config["model_name"])

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Load model
model = AutoModelForSequenceClassification.from_pretrained(config["model_name"], num_labels=2)

# Prepare training args
training_args = TrainingArguments(
    output_dir=config["output_dir"],
    eval_steps=500,
    # evaluation_strategy="steps",
    learning_rate=float(config["learning_rate"]),
    per_device_train_batch_size=int(config["batch_size"]),
    per_device_eval_batch_size=int(config["batch_size"]),
    num_train_epochs=int(config["epochs"]),
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()
trainer.save_model(config["output_dir"])
tokenizer.save_pretrained(config["output_dir"])

print("Training complete! Model saved.")
