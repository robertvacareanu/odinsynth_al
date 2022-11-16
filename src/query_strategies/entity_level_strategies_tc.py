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
    - return 'B-LOC I-LOC I-LOC' for 'New York City'
"""
import random
from scipy.stats import entropy
from typing import List


"""
In this query implementation we just select random
"""
def random_query(predictions: List[List[List[float]]], k=5, **kwargs) -> List[int]:
    return random.sample(list(range(len(predictions))), k=k)


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
    
    
    entropies_and_sentence_ids_sorted = sorted(entropies_and_sentence_ids, key=lambda x: x[2], reverse=True)
    
    # These are the selections to be annotated
    # A list of (sentence_id, token_position)
    selection = [(x[0], x[1]) for x in entropies_and_sentence_ids_sorted[:k]]
    

    entropies_and_indices = list(zip(range(len(entropies)), entropies))
    return [x[0] for x in sorted(entropies_and_indices, key=lambda x: x[1], reverse=True)[:k]]


"""
In this query implementation we select the top `k` by difference
between top two predictions
"""
def breaking_ties_query(predictions: List[List[List[float]]], k=5, **kwargs) -> List[int]:
    margins = [[sorted(y, reverse=True)[:2] for y in x] for x in predictions]
    margins = [min([y[0] - y[1] for y in x]) for x in margins]

    margins_and_indices = list(zip(range(len(margins)), margins))

    sorted_margins_and_indices = sorted(margins_and_indices, key=lambda x: x[1])

    return [x[0] for x in sorted_margins_and_indices[:k]]


def least_confidence_query(predictions: List[List[List[float]]], k=5, **kwargs) -> List[int]:
    prediction_confidence = [min([sorted(y, reverse=True)[-1] for y in x]) for x in predictions]

    prediction_confidence_and_indices = list(zip(range(len(prediction_confidence)), prediction_confidence))
    sorted_prediction_confidence_and_indices = sorted(prediction_confidence_and_indices, key=lambda x: x[1])

    return [x[0] for x in sorted_prediction_confidence_and_indices[:k]]


"""
Little test
"""
if __name__ == "__main__":
    predictions = [
        [
            # Tokens in sentence 1
            [0.99, 0.01, 0.0, 0.0], [0.98, 0.01, 0.01, 0.0], [0.5, 0.4, 0.05, 0.05], [0.5, 0.4, 0.05, 0.05], [0.97, 0.01, 0.01, 0.01], [0.96, 0.02, 0.01, 0.01], [0.95, 0.03, 0.01, 0.01], [0.94, 0.04, 0.01, 0.01],
        ],
        [
            # Tokens in sentence 2
            [0.99, 0.01, 0.0, 0.0], [0.98, 0.01, 0.01, 0.0], [0.45, 0.44, 0.11, 0.0], [0.5, 0.4, 0.05, 0.05], [0.97, 0.01, 0.01, 0.01], [0.96, 0.02, 0.01, 0.01], [0.95, 0.03, 0.01, 0.01], [0.94, 0.04, 0.01, 0.01],
        ],
        [
            # Tokens in sentence 3
            [0.99, 0.01, 0.0, 0.0], [0.98, 0.01, 0.01, 0.0], [0.97, 0.01, 0.01, 0.01], [0.96, 0.02, 0.01, 0.01], [0.95, 0.03, 0.01, 0.01], [0.94, 0.04, 0.01, 0.01],
        ],
        [
            # Tokens in sentence 4
            [0.99, 0.01, 0.0, 0.0], [0.98, 0.01, 0.01, 0.0], [0.97, 0.01, 0.01, 0.01], [0.96, 0.02, 0.01, 0.01], [0.95, 0.03, 0.01, 0.01], [0.94, 0.04, 0.01, 0.01],
        ],
        [
            # Tokens in sentence 5
            [0.99, 0.01, 0.0, 0.0], [0.98, 0.01, 0.01, 0.0], [0.97, 0.01, 0.01, 0.01], [0.96, 0.02, 0.01, 0.01], [0.95, 0.03, 0.01, 0.01], [0.94, 0.04, 0.01, 0.01],
        ],
    ]
    print(breaking_ties_query(predictions, k=2))
    print(random_query(predictions, k=2))
    print(prediction_entropy_query(predictions, k=2))
