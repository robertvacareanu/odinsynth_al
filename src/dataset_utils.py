"""
Some utils to work with the datasets in the AL scenario
"""

from datasets import load_dataset
import random

"""
Wrap the conll2003 info here
Needed because, besides the `load_dataset` call,
we might want to know the id to label mapping (i.e. `O` -> 0, etc)
"""
def get_conll2003():
    conll2003 = load_dataset("conll2003")
    label_to_id = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7, 'I-MISC': 8}
    id_to_label = {v:k for (k,v) in label_to_id.items()}
    return (conll2003, label_to_id, id_to_label)


"""
Given a dataset split, select random points
"""
def select_randomly_from_dataset(dataset, ratio=0.01):
    selected_indices = random.sample(range(0, len(dataset['train'])), int(len(dataset['train']) * ratio))
    return dataset.select(selected_indices)

