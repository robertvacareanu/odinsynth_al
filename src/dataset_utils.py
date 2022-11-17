"""
Some utils to work with the datasets in the AL scenario
"""

from datasets import load_dataset
import random

"""
Wrap the conll2003 info here
Needed because, besides the `load_dataset` call,
we might want to know the id to label mapping (i.e. `O` -> 0, etc) etc
"""
def get_conll2003():
    conll2003 = load_dataset("conll2003")

    # Just for completeness, in case we might end up using them
    postag_to_id = {'"': 0, "''": 1, '#': 2, '$': 3, '(': 4, ')': 5, ',': 6, '.': 7, ':': 8, '``': 9, 'CC': 10, 'CD': 11, 'DT': 12,
        'EX': 13, 'FW': 14, 'IN': 15, 'JJ': 16, 'JJR': 17, 'JJS': 18, 'LS': 19, 'MD': 20, 'NN': 21, 'NNP': 22, 'NNPS': 23,
        'NNS': 24, 'NN|SYM': 25, 'PDT': 26, 'POS': 27, 'PRP': 28, 'PRP$': 29, 'RB': 30, 'RBR': 31, 'RBS': 32, 'RP': 33,
        'SYM': 34, 'TO': 35, 'UH': 36, 'VB': 37, 'VBD': 38, 'VBG': 39, 'VBN': 40, 'VBP': 41, 'VBZ': 42, 'WDT': 43,
        'WP': 44, 'WP$': 45, 'WRB': 46
    }
    chunk_to_id = {'O': 0, 'B-ADJP': 1, 'I-ADJP': 2, 'B-ADVP': 3, 'I-ADVP': 4, 'B-CONJP': 5, 'I-CONJP': 6, 'B-INTJ': 7, 'I-INTJ': 8,
        'B-LST': 9, 'I-LST': 10, 'B-NP': 11, 'I-NP': 12, 'B-PP': 13, 'I-PP': 14, 'B-PRT': 15, 'I-PRT': 16, 'B-SBAR': 17,
        'I-SBAR': 18, 'B-UCP': 19, 'I-UCP': 20, 'B-VP': 21, 'I-VP': 22
    }
    
    label_to_id = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7, 'I-MISC': 8}
    id_to_label = {v:k for (k,v) in label_to_id.items()}
    return (conll2003, label_to_id, id_to_label)


"""
Given a dataset split, select random points
"""
def select_randomly_from_dataset(dataset, ratio=0.01):
    selected_indices = random.sample(range(0, len(dataset['train'])), int(len(dataset['train']) * ratio))
    return dataset.select(selected_indices)

