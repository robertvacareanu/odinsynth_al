from dataclasses import dataclass
import re
from typing import Any, Dict, List
import evaluate
import numpy as np
import torch
import random


"""
The following class is from https://github.com/thunlp/Few-NERD/blob/83299be7319092b2dc17f5d597b06d6f7aea2291/util/metric.py
No adaptation (except some new line removal)
"""
class Metrics():

    def __init__(self, ignore_index=-100):
        '''
        word_encoder: Sentence encoder
        
        You need to set self.cost as your own loss function.
        '''
        self.ignore_index = ignore_index

    def __get_class_span_dict__(self, label, is_string=False):
        '''
        return a dictionary of each class label/tag corresponding to the entity positions in the sentence
        {label:[(start_pos, end_pos), ...]}
        '''
        class_span = {}
        current_label = None
        i = 0
        if not is_string:
            # having labels in [0, num_of_class] 
            while i < len(label):
                if label[i] > 0:
                    start = i
                    current_label = label[i]
                    i += 1
                    while i < len(label) and label[i] == current_label:
                        i += 1
                    if current_label in class_span:
                        class_span[current_label].append((start, i))
                    else:
                        class_span[current_label] = [(start, i)]
                else:
                    assert label[i] == 0
                    i += 1
        else:
            # having tags in string format ['O', 'O', 'person-xxx', ..]
            while i < len(label):
                if label[i] != 'O':
                    start = i
                    current_label = label[i]
                    i += 1
                    while i < len(label) and label[i] == current_label:
                        i += 1
                    if current_label in class_span:
                        class_span[current_label].append((start, i))
                    else:
                        class_span[current_label] = [(start, i)]
                else:
                    i += 1
        return class_span

    def __get_intersect_by_entity__(self, pred_class_span, label_class_span):
        '''
        return the count of correct entity
        '''
        cnt = 0
        for label in label_class_span:
            cnt += len(list(set(label_class_span[label]).intersection(set(pred_class_span.get(label,[])))))
        return cnt
        
    def __get_cnt__(self, label_class_span):
        '''
        return the count of entities
        '''
        cnt = 0
        for label in label_class_span:
            cnt += len(label_class_span[label])
        return cnt

    def __get_correct_span__(self, pred_span, label_span):
        '''
        return count of correct entity spans
        '''
        pred_span_list = []
        label_span_list = []
        for pred in pred_span:
            pred_span_list += pred_span[pred]
        for label in label_span:
            label_span_list += label_span[label]
        return len(list(set(pred_span_list).intersection(set(label_span_list))))

    def __get_wrong_within_span__(self, pred_span, label_span):
        '''
        return count of entities with correct span, correct coarse type but wrong finegrained type
        '''
        cnt = 0
        for label in label_span:
            coarse = label.split('-')[0]
            within_pred_span = []
            for pred in pred_span:
                if pred != label and pred.split('-')[0] == coarse:
                    within_pred_span += pred_span[pred]
            cnt += len(list(set(label_span[label]).intersection(set(within_pred_span))))
        return cnt

    def __get_wrong_outer_span__(self, pred_span, label_span):
        '''
        return count of entities with correct span but wrong coarse type
        '''
        cnt = 0
        for label in label_span:
            coarse = label.split('-')[0]
            outer_pred_span = []
            for pred in pred_span:
                if pred != label and pred.split('-')[0] != coarse:
                    outer_pred_span += pred_span[pred]
            cnt += len(list(set(label_span[label]).intersection(set(outer_pred_span))))
        return cnt

    def __get_type_error__(self, pred, label, query):
        '''
        return finegrained type error cnt, coarse type error cnt and total correct span count
        '''
        pred_tag, label_tag = self.__transform_label_to_tag__(pred, query)
        pred_span = self.__get_class_span_dict__(pred_tag, is_string=True)
        label_span = self.__get_class_span_dict__(label_tag, is_string=True)
        total_correct_span = self.__get_correct_span__(pred_span, label_span) + 1e-6
        wrong_within_span = self.__get_wrong_within_span__(pred_span, label_span)
        wrong_outer_span = self.__get_wrong_outer_span__(pred_span, label_span)
        return wrong_within_span, wrong_outer_span, total_correct_span

    def metrics_by_entity_(self, pred, label):
        '''
        return entity level count of total prediction, true labels, and correct prediction
        '''
        pred_class_span = self.__get_class_span_dict__(pred, is_string=True)
        label_class_span = self.__get_class_span_dict__(label, is_string=True)
        pred_cnt = self.__get_cnt__(pred_class_span)
        label_cnt = self.__get_cnt__(label_class_span)
        correct_cnt = self.__get_intersect_by_entity__(pred_class_span, label_class_span)
        return pred_cnt, label_cnt, correct_cnt

    def metrics_by_entity(self, pred, label):
        pred_cnt = 0
        label_cnt = 0
        correct_cnt = 0
        for i in range(len(pred)):
            p_cnt, l_cnt, c_cnt = self.metrics_by_entity_(pred[i], label[i])
            pred_cnt += p_cnt
            label_cnt += l_cnt
            correct_cnt += c_cnt
        precision = correct_cnt / (pred_cnt + 1e-8)
        recall = correct_cnt / (label_cnt + 1e-8)
        f1 = 2 * precision * recall / (precision + recall)
        return precision, recall, f1



"""
Adapted from: https://huggingface.co/docs/transformers/tasks/token_classification

>>> sentences = [["the", "moon", "shines", "over", "the", "lakecity"]]
>>> tokenized_inputs = tok(sentences, truncation=True, is_split_into_words=True)
>>> tokenized_inputs
{'input_ids': [[101, 1103, 5907, 18978, 1116, 1166, 1103, 3521, 9041, 102]], 'token_type_ids': [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]}
>>> tokenized_inputs.word_ids(batch_index=0)
[None, 0, 1, 2, 2, 3, 4, 5, 5, None]
"""
def tokenize_and_align_labels(tokenizer, examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples[f"ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.

        # Memorize the previous word id for checking
        # If current word has the same id as the previous word we add `-100` to labels
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def compute_metrics(predictions, labels, id_to_label, metric=evaluate.load("seqeval"), verbose=False):
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [id_to_label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id_to_label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    
    if verbose:
        return results
    else:
        return {
            "overall_precision": results["overall_precision"],
            "overall_recall": results["overall_recall"],
            "overall_f1": results["overall_f1"],
            "overall_accuracy": results["overall_accuracy"],
        }

def compute_metrics_fewnerd(predictions, labels, id_to_label, metric=Metrics(), verbose=False):
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [id_to_label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id_to_label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    p, r, f1 = metric.metrics_by_entity(true_predictions, true_labels)

    return {
        'overall_precision': p,
        'overall_recall': r,
        'overall_f1': f1,
    }

def init_random(seed):
    """
    Init torch, torch.cuda and numpy with same seed. For reproducibility.
    :param seed: Random number generator seed
    """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

"""
Some performance printing
Mostly to check the trends
"""
def verbose_performance_printing(predictions, ali):
    print("############")
    print(f"AL Iteration: {ali}")
    print("-------Metrics-------")
    print("Overall")
    print('Overall Precision:', predictions.metrics['test_overall_precision'])
    print('Overall Recall:   ', predictions.metrics['test_overall_recall'])
    print('Overall F1:       ', predictions.metrics['test_overall_f1'])
    if 'test_overall_accuracy' in predictions.metrics:
        print('Overall Accuracy  ', predictions.metrics['test_overall_accuracy'])
    print("Per Tag")
    tags = [x[5:].upper() for x in predictions.metrics.keys() if x not in ['test_loss', 'test_overall_precision', 'test_overall_recall', 'test_overall_f1', 'test_overall_accuracy', 'test_runtime', 'test_samples_per_second', 'test_steps_per_second']]
    for tag in tags:
        scores_per_tag = predictions.metrics[f'test_{tag}']
        print(f'{tag} - {scores_per_tag}')
    print("-------Metrics-------")
    print("############")
    print("\n\n")



"""
Used for partially-annotated data points

Wrap some of the annotations we keep around in a single class
The intended utilization of this is as follows:
We maintain a data structure of objects of this class
At the end of each active learning iteration, when we query
for new data points, we also give this data structure. 
When we add new data points we check to see if those were
already partially annotated. If they were, we append the annotations
If not, we create a new object

The reason to use this instead of simply the `ner_tags` is that
maybe we will change the functionality in the future and use
the part-of-speech tags to select the tokens to be annotated
"""
@dataclass
class ALAnnotation:
    # The index of the original sentence
    sentence_id: int

    # This is the original data structure with all the labels
    # We do not (necessarily) use everything in here
    original_dict: Dict[str, Any]
    
    # Here we keep only the tags that were annotated by the user
    al_annotated_ner_tags: List[str]
    
    @staticmethod
    def from_line(line: Dict[str, Any], sid=None, al_annotated_ner_tags=None):
        if sid is None:
            sid = line['id']
        if al_annotated_ner_tags is None:
            al_annotated_ner_tags = line['ner_tags']

        return ALAnnotation(sid, line, al_annotated_ner_tags)

    """
    Return the number of ner_tags annotated
    """
    def number_of_annotated_tokens(self) -> int:
        return len([x for x in self.al_annotated_ner_tags if x != -100])

    def tokenid_selected_so_far(self) -> List[int]:
        return [x[0] for x in enumerate(self.al_annotated_ner_tags) if x[1] != -100]

    def get_annotated_tokens(self) -> List[int]:
        return [x for x in self.al_annotated_ner_tags if x != -100]

    """
    Returns one or multiple training dictionaries, according to the annotation strategy
    For example, `mask_unknown` will return the original data, but it will overwrite `ner_tags`
    to keep only those that were annotated. Everything else will be `-100`.

    annotation_strategy:
        - `mask_all_unknown`             -> All unlabeled tokens receive a -100 ner tag
        - `drop_all_unknown`             -> All unlabeled tokens are simply dropped
        - `mask_entity_looking_unknowns` -> Everything that looks like an entity (i.e. `NNP+ (IN NNP+)?`) receives -100 ner tag
        - `drop_entity_looking_unknowns` -> Everything that looks like an entity (i.e. `NNP+ (IN NNP+)?`) is dropped
        - `dynamic_window`               -> Every match will have its own sentence, extending as far as possible to the left/right
    
    As an implementation strategy, we map every tag that is not NNP or IN to `O`. We map every `NNP` to `N` and ever `IN` to I
    This is only done to make working with regex simple
    The rule then becomes: `(N+(?:IN+)?)` (N -> NNP, I -> IN). We use `re` from python
    and replace part-of-speech tags wtih single letter. Everything not in the pattern is
    replaced with `O`.
        
    """
    def get_training_annotations(self, annotation_strategy: str) -> List[Dict[str, Any]]:
        # If everything is annotated
        if all([x != -100 for x in self.al_annotated_ner_tags]):
            output_dict = {
                **self.original_dict,
                'ner_tags': self.al_annotated_ner_tags,
            }
            return [output_dict]

        

        # After this point, it means that not everything is annotated

        # Everything that is unknown is masked
        if annotation_strategy   == 'mask_all_unknown':
            output_dict = {
                **self.original_dict,
                'ner_tags': self.al_annotated_ner_tags,
            }
            return [output_dict]
        # Everything that is unknown is simply removed
        elif annotation_strategy == 'drop_all_unknown':
            output_dict = {
                **self.original_dict,
                'tokens': [],
                'ner_tags': [],
            }
            for (token, nnt) in zip(self.original_dict['tokens'], self.al_annotated_ner_tags):
                if nnt != -100:
                    output_dict['tokens'].append(token)
                    output_dict['ner_tags'].append(nnt)
            
            if len(output_dict['tokens']) == 0 or len(output_dict['ner_tags']) == 0:
                raise ValueError("Something unannotated is in the annotated list. Is everything ok?")
            return [output_dict]
        # Things that look like entities and are unannotated are masked
        elif annotation_strategy == 'mask_entity_looking_unknowns':
            pos_tags = []
            for tag in self.original_dict['pos_tags_text']:
                if tag == 'NNP':
                    pos_tags.append('N')
                elif tag == 'IN':
                    pos_tags.append('I')
                else:
                    pos_tags.append('O')

            pos_tags = ''.join(pos_tags)
            pos_tag_pattern = "(N+(?:IN+)?)"
            matches = list(re.finditer(pos_tag_pattern, pos_tags))
            
            output = []
            
            # Everything is `O` in the beginning
            ner_tags = [0] * len(self.al_annotated_ner_tags)
            
            # We iterate over all annotations and overwrite
            # if it is not -100
            for i, nt in enumerate(self.al_annotated_ner_tags):
                if nt != -100:
                    ner_tags[i] = nt

            # We iterate over each match
            # Then, we overwrite the ner_tags
            for m in matches:
                for i in range(m.start(), m.end()):
                    ner_tags[i] = self.al_annotated_ner_tags[i]
            
            output.append({
                **self.original_dict,
                'ner_tags': ner_tags,
            })

            return output
        # Things that look like entities and are unannotated are dropped
        elif annotation_strategy == 'drop_entity_looking_unknowns':
            pos_tags = []
            for tag in self.original_dict['pos_tags_text']:
                if tag == 'NNP':
                    pos_tags.append('N')
                elif tag == 'IN':
                    pos_tags.append('I')
                else:
                    pos_tags.append('O')

            pos_tags = ''.join(pos_tags)
            pos_tag_pattern = "(N+(?:IN+)?)"
            matches = list(re.finditer(pos_tag_pattern, pos_tags))
            
            output = []
            
            # We iterate over each match
            # Then, if a something that looks like a ner is
            # present, we add it
            # Otherwise, we drop it
            tokens   = []
            ner_tags = []
            matches_indices = set()
            for m in matches:
                for i in range(m.start(), m.end()):
                    matches_indices.add(i)
            
            # We iterate over ecah token
            # If that token is inside the matches, we add it only if it is annotated (i.e. its label is not `-100`)
            # Otherwise, we drop it
            for (i, token, al_annotated_ner_tag) in zip(range(len(self.al_annotated_ner_tags)), self.original_dict['tokens'], self.al_annotated_ner_tags):
                if i in matches_indices:
                    if al_annotated_ner_tag != -100:
                        tokens.append(token)
                        ner_tags.append(al_annotated_ner_tag)
                else:
                    tokens.append(token)
                    if al_annotated_ner_tag == -100:
                        ner_tags.append(0)
                    else:
                        ner_tags.append(al_annotated_ner_tag)

            output.append({
                **self.original_dict,
                'tokens'  : tokens,
                'ner_tags': ner_tags,
            })

            return output
        # We use a new sentence for each entity candidate
        elif annotation_strategy == 'dynamic_window':
            pos_tags = []
            for tag in self.original_dict['pos_tags_text']:
                if tag == 'NNP':
                    pos_tags.append('N')
                elif tag == 'IN':
                    pos_tags.append('I')
                else:
                    pos_tags.append('O')

            pos_tags = ''.join(pos_tags)
            pos_tag_pattern = "(N+(?:IN+)?)"
            matches = list(re.finditer(pos_tag_pattern, pos_tags))

            if len(matches) == 0:
                return [{
                    **self.original_dict,
                    'ner_tags': self.al_annotated_ner_tags,
                }]

            output = []
            # If there is only one match (or every entity candidate was annotated), we return the full sentence, only with that entity masked
            if len(matches) == 1 or all([self.al_annotated_ner_tags[x] != -100 for x in [j for i in matches for j in range(i.start(), i.end())]]):
                m = matches[0]
                ner_tags = [0] * len(self.al_annotated_ner_tags)

                # We iterate over all annotations and overwrite
                # if it is not -100
                for i, nt in enumerate(self.al_annotated_ner_tags):
                    if nt != -100:
                        ner_tags[i] = nt
                
                for i in range(m.start(), m.end()):
                    if self.al_annotated_ner_tags[i] != -100:
                        ner_tags[i] = self.al_annotated_ner_tags[i]            

                output.append({
                    **self.original_dict,
                    'ner_tags': ner_tags,
                })
                
            # If there are more matches, we will have as many sentences as matches
            # For each match, we extend as far as possible to the left and to the right until
            # we hit a new thing that looks like a named entity (i.e. a match)
            else:
                for i, m in enumerate(matches):
                    match_indices = set(list(range(m.start(), m.end())))
                    leftmost_index  = None
                    rightmost_index = None

                    # When this is the first or the last match
                    # we can easily set one boundary (i.e. either start or end)
                    if i == 0:
                        leftmost_index = 0
                    if i == len(matches) - 1:
                        rightmost_index = len(self.al_annotated_ner_tags)

                    # If a boundary is not set it means we were not at the first/last match

                    # Set leftmost boundary. Go step by step through each match. If a match is completely annotated,
                    # we go to the next one (i.e. to the left)
                    if leftmost_index is None:
                        j = i-1
                        while j >= 0 and all([self.al_annotated_ner_tags[x] != -100 for x in range(matches[j].start(), matches[j].end())]):
                            match_indices = match_indices.union(list(range(matches[j].start(), matches[j].end())))
                            j = j-1
                        if j < 0:
                            leftmost_index = 0
                        else:
                            leftmost_index = matches[j].end()

                    # Set rightmost boundary. Go step by step through each match. If a match is completely annotated,
                    # we go to the next one (i.e. to the right)
                    if rightmost_index is None:
                        j = i+1
                        while j <= len(matches) - 1 and all([self.al_annotated_ner_tags[x] != -100 for x in range(matches[j].start(), matches[j].end())]):
                            match_indices = match_indices.union(list(range(matches[j].start(), matches[j].end())))
                            j = j+1
                        if j > len(matches) - 1:
                            rightmost_index = len(self.al_annotated_ner_tags)
                        else:
                            rightmost_index = matches[j].start()

                    ner_tags = [0] * (rightmost_index - leftmost_index)
                    tokens   = []
                    for j in range(leftmost_index, rightmost_index):
                        tokens.append(self.original_dict['tokens'][j])
                        # If we are adding the token of the match, we respect the annotated
                        # values, even if they are `-100` (i.e. they are missing)
                        if j in match_indices:
                            ner_tags[j-leftmost_index] = self.al_annotated_ner_tags[j]
                        elif self.al_annotated_ner_tags[j] != -100:
                            ner_tags[j-leftmost_index] = self.al_annotated_ner_tags[j]

                    output.append({
                        **self.original_dict,
                        'tokens'  : tokens,
                        'ner_tags': ner_tags,
                    })
                
            return output
        else:
            raise ValueError("Unknown annotation strategy. Is everything ok?")


