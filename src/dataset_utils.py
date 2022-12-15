"""
Some utils to work with the datasets in the AL scenario
NOTE We sample a validation partition from train (1%). We
     use a static seed for that; This is because we do not
     want to introduce any more variance with the choice of 
     seed; This would simmulate how real datasets are, though
     as the `train`, `val`, and `test` partitions are static
"""

from datasets import load_dataset, DatasetDict, Dataset
import random

"""
Wrap the conll2003 info here
Needed because, besides the `load_dataset` call,
we might want to know the id to label mapping (i.e. `O` -> 0, etc) etc
train: Dataset({
    features: ['id', 'tokens', 'pos_tags', 'pos_tags_text', 'ner_tags'],
    num_rows: 13900
})
>>> len([y for x in conll2003[0]['train'] for y in x['tokens']])
201943
"""
def get_conll2003(args):
    conll2003 = load_dataset("conll2003")

    # Just for completeness, in case we might end up using them
    postag_to_id = {'"': 0, "''": 1, '#': 2, '$': 3, '(': 4, ')': 5, ',': 6, '.': 7, ':': 8, '``': 9, 'CC': 10, 'CD': 11, 'DT': 12,
        'EX': 13, 'FW': 14, 'IN': 15, 'JJ': 16, 'JJR': 17, 'JJS': 18, 'LS': 19, 'MD': 20, 'NN': 21, 'NNP': 22, 'NNPS': 23,
        'NNS': 24, 'NN|SYM': 25, 'PDT': 26, 'POS': 27, 'PRP': 28, 'PRP$': 29, 'RB': 30, 'RBR': 31, 'RBS': 32, 'RP': 33,
        'SYM': 34, 'TO': 35, 'UH': 36, 'VB': 37, 'VBD': 38, 'VBG': 39, 'VBN': 40, 'VBP': 41, 'VBZ': 42, 'WDT': 43,
        'WP': 44, 'WP$': 45, 'WRB': 46
    }
    id_to_postag = {v:k for (k,v) in postag_to_id.items()}
    
    chunk_to_id = {'O': 0, 'B-ADJP': 1, 'I-ADJP': 2, 'B-ADVP': 3, 'I-ADVP': 4, 'B-CONJP': 5, 'I-CONJP': 6, 'B-INTJ': 7, 'I-INTJ': 8,
        'B-LST': 9, 'I-LST': 10, 'B-NP': 11, 'I-NP': 12, 'B-PP': 13, 'I-PP': 14, 'B-PRT': 15, 'I-PRT': 16, 'B-SBAR': 17,
        'I-SBAR': 18, 'B-UCP': 19, 'I-UCP': 20, 'B-VP': 21, 'I-VP': 22
    }
    
    label_to_id = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7, 'I-MISC': 8}
    id_to_label = {v:k for (k,v) in label_to_id.items()}


    data_train     = []
    data_val_train = []
    data_val       = []
    data_test      = []


    # If we use the full dataset we use the full validation too; The `validation` one, not the sampled one
    if args['use_full_dataset']:
        conll2003_train_test = DatasetDict({
            'train': conll2003['train'],
            'test': conll2003['validation'],
        })
    else:
        conll2003_train_test = conll2003['train'].train_test_split(seed=1, test_size=0.01)
    

    for data, destination in zip([conll2003_train_test['train'], conll2003_train_test['test'], conll2003['validation'], conll2003['test']], [data_train, data_val_train, data_val, data_test]):
        i = 0
        for line in data:
            o = {
                'id'           : i,
                'tokens'       : line['tokens'],
                'pos_tags'     : line['pos_tags'],
                'pos_tags_text': [id_to_postag[x] for x in line['pos_tags']],
                'ner_tags'     : line['ner_tags'],
            }
            i += 1
            destination.append(o)
            

    dataset = DatasetDict({
        'train'     : Dataset.from_list(data_train),
        'val_train' : Dataset.from_list(data_val_train),
        'validation': Dataset.from_list(data_val),
        'test'      : Dataset.from_list(data_test),

    })

    return (dataset, label_to_id, id_to_label)

"""
Wrap ontonotes
train: Dataset({
    features: ['id', 'tokens', 'pos_tags', 'pos_tags_text', 'ner_tags'],
    num_rows: 74552
})
>>> len([y for x in onto[0]['train'] for y in x['tokens']])
1285509
"""
def get_ontonotes(args):
    conll2012 = load_dataset('conll2012_ontonotesv5', 'english_v4')

    postags = ["XX", "``", "$", "''", ",", "-LRB-", "-RRB-", ".", ":", "ADD", "AFX", "CC", "CD", "DT", "EX", "FW", "HYPH", "IN", "JJ", "JJR", "JJS", "LS", "MD", "NFP", "NN", "NNP", "NNPS", "NNS", "PDT", "POS", "PRP", "PRP$", "RB", "RBR", "RBS", "RP", "SYM", "TO", "UH", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "WDT", "WP", "WP$", "WRB",]
    id_to_postag = {i:k for (i, k) in enumerate(postags)}
    postag_to_id = {v:k for (k, v) in id_to_postag.items()}


    data_train     = []
    data_val_train = []
    data_val       = []
    data_test      = []

    # If we use the full dataset we use the full validation too; The `validation` one, not the sampled one
    if args['use_full_dataset']:
        conll2012_train_test = DatasetDict({
            'train': conll2012['train'],
            'test': conll2012['validation'],
        })
    else:
        conll2012_train_test = conll2012['train'].train_test_split(seed=1, test_size=0.01)

    for data, destination in zip([conll2012_train_test['train'], conll2012_train_test['test'], conll2012['validation'], conll2012['test']], [data_train, data_val_train, data_val, data_test]):
        i = 0
        for line in data:
            for s in line['sentences']:
                o = {
                    'id'           : i,
                    'tokens'       : s['words'],
                    'pos_tags'     : s['pos_tags'],
                    'pos_tags_text': [id_to_postag[x] for x in s['pos_tags']],
                    'ner_tags'     : s['named_entities'],
                }
                i += 1
                destination.append(o)



    labels = ["O", "B-PERSON", "I-PERSON", "B-NORP", "I-NORP", "B-FAC", "I-FAC", "B-ORG", "I-ORG", "B-GPE", "I-GPE", "B-LOC", "I-LOC", "B-PRODUCT", "I-PRODUCT", "B-DATE", "I-DATE", "B-TIME", "I-TIME", "B-PERCENT", "I-PERCENT", "B-MONEY", "I-MONEY", "B-QUANTITY", "I-QUANTITY", "B-ORDINAL", "I-ORDINAL", "B-CARDINAL", "I-CARDINAL", "B-EVENT", "I-EVENT", "B-WORK_OF_ART", "I-WORK_OF_ART", "B-LAW", "I-LAW", "B-LANGUAGE", "I-LANGUAGE"]
    id_to_label = {i:k for (i, k) in enumerate(labels)}
    label_to_id = {v:k for (k, v) in id_to_label.items()}

    dataset = DatasetDict({
        'train'     : Dataset.from_list(data_train),
        'val_train' : Dataset.from_list(data_val_train),
        'validation': Dataset.from_list(data_val),
        'test'      : Dataset.from_list(data_test),

    })

    return (dataset, label_to_id, id_to_label)


"""
Wrap FewNERD coarse-grained
train: Dataset({
    features: ['id', 'tokens', 'ner_tags'],
    num_rows: 130449
})
>>> len([y for x in fcg[0]['train'] for y in x['tokens']])
3195564
"""
def get_fewnerd_cg(args):
    fewnerd = load_dataset('DFKI-SLT/few-nerd', 'supervised')
    # Just for completeness, in case we might end up using them


    labels = [f'B-{x.upper()}' if x != 'O' else x.upper() for x in fewnerd['train'].features['ner_tags'].feature.names]
    id_to_label = {i:k for (i, k) in enumerate(labels)}
    label_to_id = {v:k for (k, v) in id_to_label.items()}


    data_train     = []
    data_val_train = []
    data_val       = []
    data_test      = []

    # If we use the full dataset we use the full validation too; The `validation` one, not the sampled one
    if args['use_full_dataset']:
        fewnerd_train_test = DatasetDict({
            'train': fewnerd['train'],
            'test': fewnerd['validation'],
        })
    else:
        fewnerd_train_test = fewnerd['train'].train_test_split(seed=1, test_size=0.01)

    for data, destination in zip([fewnerd_train_test['train'], fewnerd_train_test['test'], fewnerd['validation'], fewnerd['test']], [data_train, data_val_train, data_val, data_test]):
        i = 0
        for line in data:
            o = {
                'id'           : i,
                'tokens'       : line['tokens'],
                'ner_tags'     : line['ner_tags'],
            }
            i += 1
            destination.append(o)


    dataset = DatasetDict({
        'train'     : Dataset.from_list(data_train),
        'val_train' : Dataset.from_list(data_val_train),
        'validation': Dataset.from_list(data_val),
        'test'      : Dataset.from_list(data_test),

    })

    return (dataset, label_to_id, id_to_label)

"""
Wrap FewNERD fine-grained
train: Dataset({
    features: ['id', 'tokens', 'ner_tags'],
    num_rows: 130449
})
>>> len([y for x in ffg[0]['train'] for y in x['tokens']])
3195564
"""
def get_fewnerd_fg(args):
    fewnerd = load_dataset('DFKI-SLT/few-nerd', 'supervised')
    # Just for completeness, in case we might end up using them


    labels = [f'B-{x.upper()}' if x != 'O' else x.upper() for x in fewnerd['train'].features['fine_ner_tags'].feature.names]
    id_to_label = {i:k for (i, k) in enumerate(labels)}
    label_to_id = {v:k for (k, v) in id_to_label.items()}

    data_train     = []
    data_val_train = []
    data_val       = []
    data_test      = []

    # If we use the full dataset we use the full validation too; The `validation` one, not the sampled one
    if args['use_full_dataset']:
        fewnerd_train_test = DatasetDict({
            'train': fewnerd['train'],
            'test': fewnerd['validation'],
        })
    else:
        fewnerd_train_test = fewnerd['train'].train_test_split(seed=1, test_size=0.01)

    for data, destination in zip([fewnerd_train_test['train'], fewnerd_train_test['test'], fewnerd['validation'], fewnerd['test']], [data_train, data_val_train, data_val, data_test]):
        i = 0
        for line in data:
            o = {
                'id'           : i,
                'tokens'       : line['tokens'],
                'ner_tags'     : line['fine_ner_tags'],
            }
            i += 1
            destination.append(o)

    dataset = DatasetDict({
        'train'     : Dataset.from_list(data_train),
        'val_train' : Dataset.from_list(data_val_train),
        'validation': Dataset.from_list(data_val),
        'test'      : Dataset.from_list(data_test),

    })

    return (dataset, label_to_id, id_to_label)


