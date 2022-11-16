import scipy


"""
In general, we pad every sentence in the batch up to the same length
For sentence-level classification we will have, at the end, a tensor
of shape (batch_size, number_of_classes).
However, for token-level classification we will have a tensor 
of shape (batch_size, max_seq_length, number_of_classes)
In Active Learning, we select instances based on some metrics
over the predictions. For calculating metrics on the predictions of each token (e.g.
entropy, etc) we wish to skip the invalid tokens 

Invalid tokens: [CLS], [SEP], [PAD], every sub-word except the first one
"""
def filter_invalid_token_predictions(predictions):
    # Filter the [PAD] scores, the [CLS] scores, etc.
    predictions_without_invalids = []
    for (sentence, labels) in zip(predictions.predictions, predictions.label_ids):
        current_sentence = []
        for (token_scores, label) in zip(sentence, labels):
            if label == -100:
                continue
            else:
                current_sentence.append(scipy.special.softmax(token_scores, axis=0).tolist())

        predictions_without_invalids.append(current_sentence)
