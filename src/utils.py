from typing import List
import evaluate
import numpy as np
import torch
import random

"""
Adapted from: https://huggingface.co/docs/transformers/tasks/token_classification

>>> sentences = [["the", "moon", "shines", "over", "the", "lakecity"]]
>>> tokenized_inputs = tok(sentences, truncation=True, is_split_into_words=True)
>>> tokenized_inputs
{'input_ids': [[101, 1103, 5907, 18978, 1116, 1166, 1103, 3521, 9041, 102]], 'token_type_ids': [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]}
>>> tokenized_inputs.word_ids(batch_index=0)
[None, 0, 1, 2, 2, 3, 4, 5, 5, None]
"""
def tokenize_and_align_labels(tokenizer, examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples[f"ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.

        # Memorize the previous word id for checking
        # If current word has the same id as the previous word we add `-100` to labels
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def compute_metrics(predictions, labels, id_to_label, metric=evaluate.load("seqeval"), verbose=False):
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

def init_random(seed):
    """
    Init torch, torch.cuda and numpy with same seed. For reproducibility.
    :param seed: Random number generator seed
    """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

"""
Some performance printing
Mostly to check the trends
"""
def verbose_performance_printing(predictions, ali):
    print("############")
    print(f"AL Iteration: {ali}")
    print("-------Metrics-------")
    print("Overall")
    print('Overall Precision:', predictions.metrics['test_overall_precision'])
    print('Overall Recall:   ', predictions.metrics['test_overall_recall'])
    print('Overall F1:       ', predictions.metrics['test_overall_f1'])
    print('Overall Accuracy  ', predictions.metrics['test_overall_accuracy'])
    print("Per Tag")
    for tag in ['PER', 'LOC', 'ORG', 'MISC',]:
        scores_per_tag = predictions.metrics[f'test_{tag}']
        print(f'{tag} - {scores_per_tag}')
    print("-------Metrics-------")
    print("############")
    print("\n\n")

"""
Aim to select at least `number_of_examples_per_entity` for each entity type
"""
def select_representative_random_data(dataset, number_of_examples_per_entity) -> List[int]:
    pass
