"""
This file contains the associated code for querying
That is, given a list of datapoints, each with an 
associated predictions (here modelled as a list
of floats), we return a list of indices, corresponding
to the indices we want to select for annotations (as per AL
paradigm))
"""
import random
from scipy.stats import entropy
from typing import List


"""
In this query implementation we just select random
"""
def random_query(predictions: List[List[float]], k=5, **kwargs) -> List[int]:
    random.choices(list(range(len(predictions))), k=k)


"""
In this query implementation we select the top `k` by entropy
Higher entropy means more uncertainty
"""
def prediction_entropy_query(predictions: List[List[float]], k=5, **kwargs) -> List[int]:
    entropies = [entropy(x) for x in predictions]

    entropies_and_indices = list(zip(range(len(entropies)), entropies))
    return [x[0] for x in sorted(entropies_and_indices, key=lambda x: x[1], reverse=True)[:k]]


"""
In this query implementation we select the top `k` by difference
between top two predictions
"""
def breaking_ties_query(predictions: List[List[float]], k=5, **kwargs) -> List[int]:
    margins = [sorted(x, reverse=True)[:2] for x in predictions]
    margins = [x[0] - x[1] for x in margins]

    margins_and_indices = list(zip(range(len(margins)), margins))

    sorted_margins_and_indices = sorted(margins_and_indices, key=lambda x: x[1])

    return [x[0] for x in sorted_margins_and_indices[:k]]


def least_confidence_query(predictions: List[List[float]], k=5, **kwargs) -> List[int]:
    prediction_confidence = [sorted(x, reverse=True)[0] for x in predictions]

    prediction_confidence_and_indices = list(zip(range(len(prediction_confidence)), prediction_confidence))
    sorted_prediction_confidence_and_indices = sorted(prediction_confidence_and_indices, key=lambda x: x[1])

    return [x[0] for x in sorted_prediction_confidence_and_indices[:k]]

