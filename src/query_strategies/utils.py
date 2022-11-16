from typing import Any, List, Tuple, Union
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
    
    return predictions_without_invalids

"""
:param dataset -> something that looks like a Huggingface dataset
            each line dictionary should have, additionally,
            a field called `al_labels` which stands for 
            `active learning labels`. This is the field which 
            contains the tokens that were annotated (either initially
            or by querying the user)
:param selected_dataset_so_far -> what we annotated so far;
            we need to know this because we might select different
            spans in the same sentence; So we don't want to throw
            away old annotations
:param selections -> what to annotate
            can be either a list of int, case in which we select full
            sentences;
            or list of (int, List[int]), which means we annotate the
            token from list j for sentence i; This allows us to
            use the same function to annotate:
            (i)   a sentence, fully
            (ii)  an entity, fully
            (iii) a part of the entity only
"""
def annotate(dataset, selected_dataset_so_far: List[Tuple[int, Any]], selections: Union[List[int], List[Tuple[int, List[int]]]]) -> List:
    if len(selections) == 0:
        raise ValueError("Nothing selected. Is everything ok?")

    selected_dataset_so_far_dict = {k:v for (k, v) in selected_dataset_so_far}

    if isinstance(selections[0], int):
        sentences_id = selections
        tokens_id    = None
    elif isinstance(selections[0], Tuple):
        sentences_id = [x[0] for x in selections]
        tokens_id    = [x[1] for x in selections]
    else:
        raise ValueError("Unknown parameter type. We expect `selections` to be either `List[int]` or `List[Tuple[int, int]]`. Is everything ok?")

    selected_dataset = dataset.select(sentences_id)
    annotated_data = []
    # In this setting we select the full sentence
    # This is because we do not have any tokens_id
    if tokens_id is None:
        for sid, line in zip(sentences_id, selected_dataset):
            line_dict_copy = {**line}
            annotated_data.append((sid, line_dict_copy))
    # In this setting we select individual tokens to annotate
    # This is because we have a token_id so we know which tokens
    # we wish to select
    else:
        for sid, tid, line in zip(sentences_id, tokens_id, selected_dataset):
            line_dict_copy = {**line}
            
            line_labels = line['ner_tags']

            # If this sentence was already selected before, we get the labels we have annotated so far
            # and add the new labels to them
            # Otherwise, get a clean list of `-100`
            # Also, we get a copy of it
            if sid in selected_dataset_so_far_dict:
                labels_so_far = [*selected_dataset_so_far_dict[sid]['al_labels']]
            else:
                labels_so_far = [-100] * len(line_labels)
                
            for token_to_annotate in tid:
                labels_so_far[token_to_annotate] = line_labels[token_to_annotate]

            annotated_data.append((sid, {**line_dict_copy, 'al_labels': labels_so_far}))
        
    # Now we also have to add all the examples that were annotated before
    # We skip over the ones with the same sentence id as the ones added
    # We do so by adding to another list because we want to keep the relative order
    # Which is first examples that were already annotated, then new examples
    # NOTE: If an example is gettign additional annotations, then it will not 
    # maintain its order
    sentences_id_set = set(sentences_id)
    original_data = []
    for (original_sid, line) in selected_dataset_so_far:
        # If this annotated sentence is not already added, add it
        if original_sid not in sentences_id_set:
            original_data.append((original_sid, line))


    return original_data + annotated_data

