from select import select
import unittest

import random


from src.query_strategies.entity_level_strategies_tc import breaking_ties_bernoulli_query, random_query, prediction_entropy_query, breaking_ties_query, least_confidence_query
from src.utils import ALAnnotation

class TestEntityLevelStrategiesTokenClassification(unittest.TestCase):

    def get_data(self):
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
        ]

        dataset = []
        dataset.append({
            'id': 0,
            'tokens': [f'0{x}' for x in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']],
            'ner_tags': [0,0,0,0,0,0,0,1],
        })
        dataset.append({
            'id': 1,
            'tokens': [f'1{x}' for x in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']],
            'ner_tags': [1,0,1,0,0,0,0,1],
        })
        dataset.append({
            'id': 2,
            'tokens': [f'2{x}' for x in ['a', 'b', 'c', 'd', 'e', 'f']],
            'ner_tags': [1,0,1,2,0],
        })

        id_to_label = {0: 'O', 1: 'B-PER', 2: 'I-PER'}
        
        selected_dataset_so_far = []

        return dataset, selected_dataset_so_far, id_to_label, predictions

    def test_random_query(self):
        dataset, selected_dataset_so_far, id_to_label, predictions = self.get_data()
        
        random.seed(1)
        output = random_query(predictions, k=1, dataset_so_far=selected_dataset_so_far, dataset=dataset, id_to_label=id_to_label)

        # Check that at least one token was annotated
        self.assertGreaterEqual(output[0][1].number_of_annotated_tokens(), 1)


        
        
    def test_prediction_entropy_query(self):
        dataset, selected_dataset_so_far, id_to_label, predictions = self.get_data()
        
        output = prediction_entropy_query(predictions, k=1, dataset_so_far=selected_dataset_so_far, dataset=dataset, id_to_label=id_to_label)

        # Check that at least one token was annotated
        self.assertEqual(output[0][1].number_of_annotated_tokens(), 1)
        self.assertEqual(output[0][1].al_annotated_ner_tags[2], 0)


    def test_breaking_ties_query(self):
        dataset, selected_dataset_so_far, id_to_label, predictions = self.get_data()
        
        random.seed(1)
        output = breaking_ties_query(predictions, k=1, dataset_so_far=selected_dataset_so_far, dataset=dataset, id_to_label=id_to_label)

        # Check that at least one token was annotated
        self.assertGreaterEqual(output[0][1].number_of_annotated_tokens(), 1)
        self.assertEqual(output[0][1].al_annotated_ner_tags[2], 1)


    def test_least_confidence_query(self):
        dataset, selected_dataset_so_far, id_to_label, predictions = self.get_data()
        
        random.seed(1)
        output = least_confidence_query(predictions, k=1, dataset_so_far=selected_dataset_so_far, dataset=dataset, id_to_label=id_to_label)

        # Check that at least one token was annotated
        self.assertGreaterEqual(output[0][1].number_of_annotated_tokens(), 1)
        self.assertEqual(output[0][1].al_annotated_ner_tags[2], 1)

    def test_breaking_ties_bernoulli_query(self):
        dataset, selected_dataset_so_far, id_to_label, predictions = self.get_data()
        
        random.seed(1)
        output = breaking_ties_bernoulli_query(predictions, k=5, dataset_so_far=selected_dataset_so_far, dataset=dataset, id_to_label=id_to_label)
        print(output)

        # Check that at least one token was annotated
        self.assertGreaterEqual(output[0][1].number_of_annotated_tokens(), 1)
        self.assertEqual(output[0][1].al_annotated_ner_tags[2], 1)




if __name__ == '__main__':
    unittest.main()
