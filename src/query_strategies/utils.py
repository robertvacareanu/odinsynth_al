from typing import Any, Dict, List, Tuple, Union
from itertools import takewhile
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
            (type: List[(int, List[int])]; a list of tuples, the first 
            element is the sentence_id and the second element is 
            the active learning annotations (i.e. the ner tags,
            but maybe not all))
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
def annotate(dataset, selected_dataset_so_far: List[Tuple[int, List[int]]], selections: Union[List[int], List[Tuple[int, List[int]]]]) -> List[Tuple[int, List[int]]]:
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

    selected_dataset = [dataset[sid] for sid in sentences_id]
    annotated_data = []
    # In this setting we select the full sentence
    # This is because we do not have any tokens_id
    if tokens_id is None:
        for sid, line in zip(sentences_id, selected_dataset):
            annotated_data.append((sid, line['ner_tags']))
    # In this setting we select individual tokens to annotate
    # This is because we have a token_id so we know which tokens
    # we wish to select
    else:
        for sid, tid, line in zip(sentences_id, tokens_id, selected_dataset):            
            line_labels = line['ner_tags']

            # If this sentence was already selected before, we get the labels we have annotated so far
            # and add the new labels to them
            # Otherwise, get a clean list of `-100`
            # Also, we get a copy of it
            if sid in selected_dataset_so_far_dict:
                labels_so_far = [*selected_dataset_so_far_dict[sid]]
            else:
                labels_so_far = [-100] * len(line_labels)
                
            for token_to_annotate in tid:
                labels_so_far[token_to_annotate] = line_labels[token_to_annotate]

            annotated_data.append((sid, labels_so_far))
        
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


"""
When we operate at span-level query, we want to:
- take only that token when that token's label is `O`
- take the full entity when that token's label is not `O`
NOTE: We are making use of the label here, but only for query simulation
Realistically, the situation would be something like:
    You selected a token. You give to the user the token + a window
    If that token is part of a named entity, annotate the full named entity
    Otherwise just say `O`.


:param labels     : the gold labels
:param id_to_label: a dictionary from a label id to str (i.e. 1 -> B-PER) 
:param token_id   : the token we chose to annotate
"""
def take_full_entity(labels: List[int], id_to_label: Dict[int, str], token_id) -> List[int]:
    labels_str = [id_to_label[x] for x in labels]
    # If the gold label is 'O', we annotate only that one
    if labels_str[token_id] == 'O':
        return [token_id]
    

    # Now we know that it is not an 'O', we have to check whether it is a
    # Beginning of an entity (i.e. 'B-') or if it is inside (i.e. 'I-')
    
    token_ids = []
    token_id_label = labels_str[token_id]
    # If it is 'B-', we take to the left as long as the tokens to the left
    # are of type 'I-' and the same named entity
    if token_id_label[0:2] == 'B-':
        token_ids.append(token_id)
        output = takewhile(lambda x: x[1][:2] == 'I-' and x[1][2:] == token_id_label[2:], list(enumerate(labels_str))[(token_id+1):])
        output = list(output)
        token_ids += [x[0] for x in output]
    # If it is 'I-' it means we are in the middle of an entity
    # We take to the left and to the right
    elif token_id_label[0:2] == 'I-':
        # Take all 'I-' to the left
        output_left = takewhile(lambda x: labels_str[x[0]][:2] == 'I-' and labels_str[x[0]][2:] == token_id_label[2:], reversed(list(enumerate(labels_str))[:(token_id+1)]))
        output_left = list(reversed(list(output_left)))
        # We took all 'I-' to the left
        # This means that tbe very next token should be 'B-'. Otherwise, throw an error
        if labels_str[output_left[0][0]-1][0:2] != 'B-' and labels_str[output_left[0][0]-1][2:] != token_id_label[2:]:
            raise ValueError(f"Invalid sequence. We should have a `B-` with the same tag, but we do not. It is {labels_str}. Is everything ok?")
        # Take all 'I-' to the right
        output_right = takewhile(lambda x: x[1][:2] == 'I-' and x[1][2:] == token_id_label[2:], list(enumerate(labels_str))[(token_id+1):])
        output_right = list(output_right)
        token_ids += [output_left[0][0]-1] + [x[0] for x in output_left] + [x[0] for x in output_right]
    else:
        raise ValueError("Unknown label. It should be either `B-*` or `I-*`. Is everything ok?")


    return token_ids
