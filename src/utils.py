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

