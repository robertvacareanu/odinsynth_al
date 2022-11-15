from src.utils import tokenize_and_align_labels
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification
from transformers import AutoTokenizer
from datasets import load_dataset
import evaluate
import numpy as np

conll2003 = load_dataset("conll2003")

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

tokenized_conll2003 = conll2003.map(lambda x: tokenize_and_align_labels(tokenizer, x), batched=True)
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, return_tensors="pt")
model = AutoModelForTokenClassification.from_pretrained("distilbert-base-uncased", num_labels=14)

metric = evaluate.load("seqeval")
def compute_metrics(predictions, labels, id_to_label, verbose=False):
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [id_to_label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id_to_label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    if verbose:
        return results
    else:
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

label_to_id = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7, 'I-MISC': 8}
id_to_label = {v:k for (k,v) in label_to_id.items()}

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
    compute_metrics=lambda x: compute_metrics(x[0], x[1], id_to_label, verbose=True)
)

trainer.train()
trainer.evaluate()
