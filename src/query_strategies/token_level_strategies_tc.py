"""
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


"""
In this query implementation we just select random
"""
def random_query(predictions: List[List[List[float]]], k=5, **kwargs) -> List[int]:
    token_and_sentence_ids = []
    for sid, sentence in enumerate(predictions):
        for tid, token in enumerate(sentence):
            token_and_sentence_ids.append((sid, [tid]))
    return random.sample(list(range(len(token_and_sentence_ids))), k=k)


"""
In this query implementation we select the top `k` by entropy
Higher entropy means more uncertainty
"""
def prediction_entropy_query(predictions: List[List[List[float]]], k=5, **kwargs) -> List[int]:
    token_and_sentence_ids = []
    for sid, sentence in enumerate(predictions):
        for tid, token in enumerate(sentence):
            token_and_sentence_ids.append((sid, [tid], entropy(token)))

    # Sort by entropy in reverse
    sorted_data = sorted(token_and_sentence_ids, key=lambda x: x[2], reverse=True)

    selected_data = [(x[1], x[2]) for x in sorted_data[:k]]

    return selected_data


"""
In this query implementation we select the top `k` by difference
between top two predictions
"""
def breaking_ties_query(predictions: List[List[List[float]]], k=5, **kwargs) -> List[int]:
    token_and_sentence_ids = []
    for sid, sentence in enumerate(predictions):
        for tid, token in enumerate(sentence):
            scores = sorted(token, reverse=True)[:2]
            token_and_sentence_ids.append((sid, [tid], scores[0] - scores[1]))

    # Sort by margins
    sorted_data = sorted(token_and_sentence_ids, key=lambda x: x[2])

    selected_data = [(x[1], x[2]) for x in sorted_data[:k]]

    return selected_data


def least_confidence_query(predictions: List[List[List[float]]], k=5, **kwargs) -> List[int]:
    token_and_sentence_ids = []
    for sid, sentence in enumerate(predictions):
        for tid, token in enumerate(sentence):
            scores = sorted(token, reverse=True)
            token_and_sentence_ids.append((sid, [tid], scores[0]))

    # Sort by confidence in reverse
    sorted_data = sorted(token_and_sentence_ids, key=lambda x: x[2])

    selected_data = [(x[1], x[2]) for x in sorted_data[:k]]

    return selected_data

