"""
Query strategy at span level

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
from collections import defaultdict
import random
from scipy.stats import entropy
from typing import Any, List

from src.query_strategies.utils import annotate, take_full_entity

"""
Just ensure that the additional information we might be using
has the same length as the predictions
This is a good check to have simply because `zip` hides away
any size mismatch
"""
def sanity_check(predictions, other_label):
    # /SANITY
    # Simple sanity checks becauze zip hides away length mismatches
    if (len(predictions) != len(other_label)):
        raise ValueError("Mismatch between predictions and labels")
    for p, tl in zip(predictions, other_label):
        if (len(p) != len(tl)):
            raise ValueError("Mismatch between predictions and labels")


"""
In this query implementation we just select random
"""
def random_query(predictions: List[List[List[float]]], k=5, **kwargs) -> List[Any]:
    token_and_sentence_ids = []
    for sid, sentence in enumerate(predictions):
        for tid, token in enumerate(sentence):
            token_and_sentence_ids.append((sid, tid))

    selected_data = random.sample(token_and_sentence_ids, k=k)

    dataset = kwargs.get('dataset')
    # Collapse every selection for every sentence
    sentenceid_to_tokensid = defaultdict(list)
    for (sid, tid) in selected_data:
        expanded_tid = take_full_entity(labels=dataset[sid]['ner_tags'], id_to_label=kwargs.get('id_to_label'), token_id=tid)
        sentenceid_to_tokensid[sid] += expanded_tid
    
    return annotate(dataset=dataset, selected_dataset_so_far=kwargs.get('dataset_so_far'), selections=list(sentenceid_to_tokensid.items()))


"""
In this query implementation we select the top `k` by entropy
Higher entropy means more uncertainty
"""
def prediction_entropy_query(predictions: List[List[List[float]]], k=5, **kwargs) -> List[Any]:
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
    selected = [(x[0], x[1]) for x in entropies_and_sentence_ids_sorted[:k]]

    dataset = kwargs.get('dataset')
    # Collapse every selection for every sentence
    sentenceid_to_tokensid = defaultdict(list)
    for (sid, tid) in selected:
        expanded_tid = take_full_entity(labels=dataset[sid]['ner_tags'], id_to_label=kwargs.get('id_to_label'), token_id=tid)
        sentenceid_to_tokensid[sid] += expanded_tid
    
    return annotate(dataset=dataset, selected_dataset_so_far=kwargs.get('dataset_so_far'), selections=list(sentenceid_to_tokensid.items()))


"""
In this query implementation we select the top `k` by difference
between top two predictions
"""
def breaking_ties_query(predictions: List[List[List[float]]], k=5, **kwargs) -> List[Any]:
    token_and_sentence_ids = []
    for sid, sentence in enumerate(predictions):
        for tid, token in enumerate(sentence):
            scores = sorted(token, reverse=True)[:2]
            token_and_sentence_ids.append((sid, tid, scores[0] - scores[1]))

    # Sort by margins
    sorted_data = sorted(token_and_sentence_ids, key=lambda x: x[2])

    selected_data = [(x[0], x[1]) for x in sorted_data[:k]]

    dataset = kwargs.get('dataset')
    # Collapse every selection for every sentence
    sentenceid_to_tokensid = defaultdict(list)
    for (sid, tid) in selected_data:
        expanded_tid = take_full_entity(labels=dataset[sid]['ner_tags'], id_to_label=kwargs.get('id_to_label'), token_id=tid)
        sentenceid_to_tokensid[sid] += expanded_tid
    
    return annotate(dataset=dataset, selected_dataset_so_far=kwargs.get('dataset_so_far'), selections=list(sentenceid_to_tokensid.items()))



def least_confidence_query(predictions: List[List[List[float]]], k=5, **kwargs) -> List[Any]:
    token_and_sentence_ids = []
    for sid, sentence in enumerate(predictions):
        for tid, token in enumerate(sentence):
            scores = sorted(token, reverse=True)
            token_and_sentence_ids.append((sid, [tid], scores[0]))

    # Sort by confidence in reverse
    sorted_data = sorted(token_and_sentence_ids, key=lambda x: x[2])

    selected_data = [(x[0], x[1]) for x in sorted_data[:k]]

    dataset = kwargs.get('dataset')
    # Collapse every selection for every sentence
    sentenceid_to_tokensid = defaultdict(list)
    for (sid, tid) in selected_data:
        expanded_tid = take_full_entity(labels=dataset[sid]['ner_tags'], id_to_label=kwargs.get('id_to_label'), token_id=tid)
        sentenceid_to_tokensid[sid] += expanded_tid
    
    return annotate(dataset=dataset, selected_dataset_so_far=kwargs.get('dataset_so_far'), selections=list(sentenceid_to_tokensid.items()))





"""
Little test
"""
if __name__ == "__main__":
    import datasets
    import random
    import numpy as np

    label_to_id = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7, 'I-MISC': 8}
    i2l = {v:k for (k,v) in label_to_id.items()}

    
    np.random.seed(1)
    random.seed(1)
    
    conll2003 = datasets.load_dataset('conll2003')['train'].select(range(5))
    predictions = [
        np.random.random((len(conll2003[0]['tokens']), len(i2l))).tolist(),
        np.random.random((len(conll2003[1]['tokens']), len(i2l))).tolist(),
        np.random.random((len(conll2003[2]['tokens']), len(i2l))).tolist(),
        np.random.random((len(conll2003[3]['tokens']), len(i2l))).tolist(),
        np.random.random((len(conll2003[4]['tokens']), len(i2l))).tolist(),
    ]
    print(conll2003[0])
    btq = random_query(predictions, k=2, dataset=conll2003, dataset_so_far=[], id_to_label=i2l)
    print(btq)
