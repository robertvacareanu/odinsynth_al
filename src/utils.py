from dataclasses import dataclass
import re
from typing import Any, Dict, List
import evaluate
import numpy as np
import torch
import random

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
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
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
        - `dynamic_window`               -> 
    
    As an implementation strategy, we map every tag that is not NNP or IN to `O`. We map every `NNP` to `N` and ever `IN` to I
    This is only done to make working with regex simple
    The rule then becomes: `(N+(?:IN+)?)`
        
    """
    def get_training_annotations(self, annotation_strategy: str) -> List[Dict[str, Any]]:
        # Everything that is unknown is masked
        if annotation_strategy   == 'mask_all_unknown':
            output_dict = {
                **self.original_dict,
                'ner_tags': self.al_annotated_ner_tags,
            }
            return [output_dict]
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
            
            output = []
            # If there is only one match, we return the full sentence, only with that entity masked
            if len(matches) == 1:
                m = matches[0]
                ner_tags = []
                
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
                    if leftmost_index is None:
                        leftmost_index = matches[i-1].end()
                    if rightmost_index is None:
                        rightmost_index = matches[i+1].start()

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
