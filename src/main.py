from src.utils import tokenize_and_align_labels
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification
from transformers import AutoTokenizer
from datasets import load_dataset
import numpy as np

conll2003 = load_dataset("conll2003")

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

tokenized_conll2003 = conll2003.map(lambda x: tokenize_and_align_labels(tokenizer, x), batched=True)
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, return_tensors="pt")
model = AutoModelForTokenClassification.from_pretrained("distilbert-base-uncased", num_labels=14)


training_args = TrainingArguments(
    output_dir="./outputs",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_conll2003["train"],
    eval_dataset=tokenized_conll2003["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()
