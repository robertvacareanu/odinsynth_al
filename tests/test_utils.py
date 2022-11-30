import unittest

from src.query_strategies.utils import annotate, filter_already_selected_sidtid_pairs, take_full_entity
from datasets import load_dataset

from src.utils import ALAnnotation

class TestEntityLevelStrategiesTokenClassification(unittest.TestCase):

    def test_ALAnnotation_get_training_annotations1(self):
        line = {
            'id': 0,
            'tokens': ['Lorem', 'ipsum', 'dolor', 'sit', 'amet,', 'consectetur', 'adipiscing', 'elit,', 'sed', 'do', 'eiusmod', 'tempor', 'incididunt', 'ut', 'labore', 'et', 'dolore', 'magna', 'aliqua'],
            'pos_tags': list(range(19)),
            'pos_tags_text': ['NNP', 'O', 'O', 'NNP', 'IN', 'NNP', 'NNP', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'NNP', 'IN', 'NNP'],
            'ner_tags': list(range(19))
        }
        al_annotated_ner_tags = [-100, -100, -100, 1, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        al_annotation = ALAnnotation(0, line, al_annotated_ner_tags)
        self.assertEqual(al_annotation.get_training_annotations("mask_all_unknown")[0]['ner_tags'], al_annotated_ner_tags)
        self.assertEqual(al_annotation.get_training_annotations("drop_all_unknown")[0]['ner_tags'], [1, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(al_annotation.get_training_annotations("mask_entity_looking_unknowns")[0]['ner_tags'], [-100, 0, 0, 1, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(al_annotation.get_training_annotations("drop_entity_looking_unknowns")[0]['ner_tags'], [0, 0, 1, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(al_annotation.get_training_annotations("mask_entity_looking_unknowns")[0]['ner_tags'], [-100, 0, 0, 1, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(al_annotation.get_training_annotations("dynamic_window")[0]['ner_tags'], [-100, 0, 0, 1, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(al_annotation.get_training_annotations("dynamic_window")[1]['ner_tags'], [0, 0, 1, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(al_annotation.get_training_annotations("dynamic_window")[2]['ner_tags'], [0, 0, 1, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

if __name__ == '__main__':
    unittest.main()




