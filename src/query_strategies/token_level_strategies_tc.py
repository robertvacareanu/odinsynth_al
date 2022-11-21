"""
Query strategy at token level

This file contains the associated code for querying
That is, given a list of datapoints, each with an 
associated predictions (here modelled as a list
of floats), we return a list of indices, corresponding
to the indices we want to select for annotations (as per AL
paradigm))
Particular to this file, we model each token prediction inside a sentence,
hence the `List[List[List[float]]]` type. 
We perform the decision based on the individual token's prediction, and,
at the end, annotate only that span

In-depth explanation of the rules:
Let us assume we have the following sentence:
`John was born in New York City and works at the United Airlines`

For simplicity, let us assume that each word in the previous sentence
has a single 
The rules employed here are as follows:
- Assuming `born` was selected for annotation
    - return `O` for `born`
- Assuming `New` was selected for annotation
    - return 'B-LOC' for 'New'
"""
import random
from scipy.stats import entropy
from typing import List

from src.query_strategies.utils import annotate, collapse_same_sentenceid_tokens, filter_already_selected_sidtid_pairs


"""
In this query implementation we just select random
"""
def random_query(predictions: List[List[List[float]]], k=5, **kwargs) -> List[int]:
    sentence_and_token_ids = []
    for sid, sentence in enumerate(predictions):
        for tid, token in enumerate(sentence):
            sentence_and_token_ids.append((sid, tid))

    sentence_and_token_ids = filter_already_selected_sidtid_pairs(sentence_and_token_ids, kwargs.get('dataset_so_far'))
    selected = [(x[0], x[1]) for x in sentence_and_token_ids]

    # We already selected everything
    if len(selected) == 0:
        return []

    # These are the selections to be annotated
    # A list of (sentence_id, token_position)
    selected = collapse_same_sentenceid_tokens(random.sample(selected, k=min(k, len(selected))))

    dataset = kwargs.get('dataset')

    return annotate(dataset=dataset, selected_dataset_so_far=kwargs.get('dataset_so_far'), selections=selected)


"""
In this query implementation we select the top `k` by entropy
Higher entropy means more uncertainty
"""
def prediction_entropy_query(predictions: List[List[List[float]]], k=5, **kwargs) -> List[int]:
    # Calculate the entropy of each token prediction
    entropies = [[entropy(y) for y in x] for x in predictions]

    # Unroll everything, but keeping track of the sentence id and token id
    entropies_and_sentence_ids = []
    for sentence_id, sentence in enumerate(entropies):
        for token_pos, token_entropy in enumerate(sentence):
            entropies_and_sentence_ids.append((sentence_id, token_pos, token_entropy))
    
    
    sorted_data = sorted(entropies_and_sentence_ids, key=lambda x: x[2], reverse=True)
    
    # These are the selections to be annotated
    # A list of (sentence_id, token_position)
    sentence_and_token_ids = [(x[0], x[1]) for x in sorted_data]
    selected = collapse_same_sentenceid_tokens(filter_already_selected_sidtid_pairs(sentence_and_token_ids, kwargs.get('dataset_so_far'))[:k])

    # We already selected everything
    if len(sentence_and_token_ids) == 0:
        return []


    dataset = kwargs.get('dataset')

    return annotate(dataset=dataset, selected_dataset_so_far=kwargs.get('dataset_so_far'), selections=selected)

"""
In this query implementation we select the top `k` by difference
between top two predictions
"""
def breaking_ties_query(predictions: List[List[List[float]]], k=5, **kwargs) -> List[int]:
    token_and_sentence_ids = []
    for sid, sentence in enumerate(predictions):
        for tid, token in enumerate(sentence):
            scores = sorted(token, reverse=True)[:2]
            token_and_sentence_ids.append((sid, tid, scores[0] - scores[1]))

    # Sort by margins
    sorted_data = sorted(token_and_sentence_ids, key=lambda x: x[2])

    # These are the selections to be annotated
    # A list of (sentence_id, token_position)
    sentence_and_token_ids = [(x[0], x[1]) for x in sorted_data]
    selected = collapse_same_sentenceid_tokens(filter_already_selected_sidtid_pairs(sentence_and_token_ids, kwargs.get('dataset_so_far'))[:k])
    # print(selected)

    # We already selected everything
    if len(sentence_and_token_ids) == 0:
        return []

    dataset = kwargs.get('dataset')
    
    return annotate(dataset=dataset, selected_dataset_so_far=kwargs.get('dataset_so_far'), selections=selected)


def least_confidence_query(predictions: List[List[List[float]]], k=5, **kwargs) -> List[int]:
    token_and_sentence_ids = []
    for sid, sentence in enumerate(predictions):
        for tid, token in enumerate(sentence):
            scores = sorted(token, reverse=True)
            token_and_sentence_ids.append((sid, tid, scores[0]))

    # Sort by confidence ascending
    # We are interested in taking the ones with the lowest confidence
    sorted_data = sorted(token_and_sentence_ids, key=lambda x: x[2])

    # These are the selections to be annotated
    # A list of (sentence_id, token_position)
    sentence_and_token_ids = [(x[0], x[1]) for x in sorted_data]
    selected = collapse_same_sentenceid_tokens(filter_already_selected_sidtid_pairs(sentence_and_token_ids, kwargs.get('dataset_so_far'))[:k])

    dataset = kwargs.get('dataset')
    
    return annotate(dataset=dataset, selected_dataset_so_far=kwargs.get('dataset_so_far'), selections=selected)

