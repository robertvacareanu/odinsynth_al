"""
Query strategy at sentence level

This file contains the associated code for querying
That is, given a list of datapoints, each with an 
associated predictions (here modelled as a list
of floats), we return a list of indices, corresponding
to the indices we want to select for annotations (as per AL
paradigm))
Particular to this file, we model each token prediction inside a sentence,
hence the `List[List[List[float]]]` type. 
We perform the decision based on the individual token's prediction,
but at the end we return sentences
"""
import random
from scipy.stats import entropy
from typing import List, Tuple
from src.query_strategies.utils import annotate

from src.utils import ALAnnotation


"""
In this query implementation we just select random
"""
def random_query(predictions: List[List[List[float]]], k=5, **kwargs) -> List[Tuple[int, List[int]]]:
    dataset_so_far = dict(kwargs.get('dataset_so_far'))

    selected_indices = random.sample([x for x in list(range(len(predictions))) if x not in dataset_so_far], k=k)    
    
    dataset = kwargs.get('dataset')
    output  = []
    for si in selected_indices:
        output.append(si)

    return annotate(dataset=dataset, selected_dataset_so_far=kwargs.get('dataset_so_far'), selections=output)


"""
In this query implementation we select the top `k` by entropy
Higher entropy means more uncertainty
"""
def prediction_entropy_query(predictions: List[List[List[float]]], k=5, **kwargs) -> List[Tuple[int, List[int]]]:
    aggregation_function = kwargs.get('aggregation_function', max)
    entropies = [aggregation_function([entropy(y) for y in x]) for x in predictions]

    entropies_and_indices = list(zip(range(len(entropies)), entropies))

    dataset_so_far = dict(kwargs.get('dataset_so_far'))


    selected_indices = [x[0] for x in sorted(entropies_and_indices, key=lambda x: x[1], reverse=True) if x[0] not in dataset_so_far][:k]

    dataset = kwargs.get('dataset')
    output  = []
    for si in selected_indices:
        output.append(si)

    return annotate(dataset=dataset, selected_dataset_so_far=kwargs.get('dataset_so_far'), selections=output)



"""
In this query implementation we select the top `k` by difference
between top two predictions
"""
def breaking_ties_query(predictions: List[List[List[float]]], k=5, **kwargs) -> List[Tuple[int, List[int]]]:
    aggregation_function = kwargs.get('aggregation_function', min)
    margins = [[sorted(y, reverse=True)[:2] for y in x] for x in predictions]
    margins = [aggregation_function([y[0] - y[1] for y in x]) for x in margins]

    margins_and_indices = list(zip(range(len(margins)), margins))

    sorted_margins_and_indices = sorted(margins_and_indices, key=lambda x: x[1])

    dataset_so_far = dict(kwargs.get('dataset_so_far'))

    selected_indices = [x[0] for x in sorted_margins_and_indices if x[0] not in dataset_so_far][:k]

    dataset = kwargs.get('dataset')
    output  = []
    for si in selected_indices:
        output.append(si)
        
    return annotate(dataset=dataset, selected_dataset_so_far=kwargs.get('dataset_so_far'), selections=output)



def least_confidence_query(predictions: List[List[List[float]]], k=5, **kwargs) -> List[Tuple[int, List[int]]]:
    aggregation_function = kwargs.get('aggregation_function', min)
    prediction_confidence = [aggregation_function([sorted(y, reverse=True)[-1] for y in x]) for x in predictions]

    prediction_confidence_and_indices = list(zip(range(len(prediction_confidence)), prediction_confidence))
    sorted_prediction_confidence_and_indices = sorted(prediction_confidence_and_indices, key=lambda x: x[1])

    dataset_so_far = dict(kwargs.get('dataset_so_far'))

    selected_indices = [x[0] for x in sorted_prediction_confidence_and_indices if x[0] not in dataset_so_far][:k]
    dataset = kwargs.get('dataset')
    output  = []
    for si in selected_indices:
        output.append(si)

    return annotate(dataset=dataset, selected_dataset_so_far=kwargs.get('dataset_so_far'), selections=output)



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
