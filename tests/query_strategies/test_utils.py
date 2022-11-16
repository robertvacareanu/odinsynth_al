import unittest

from src.query_strategies.utils import annotate
from datasets import load_dataset

class TestEntityLevelStrategiesTokenClassification(unittest.TestCase):

    """
    Check token-level annotation
    """
    def test_annotate_dataset1(self):
        conll2003 = load_dataset("conll2003")['train']

        # The dataset we have selected so far
        # A list of (sentence_id, line_dictionary)
        selected_dataset_so_far = []

        # This one is partially annotated (missing annotation for 0)
        labels0 = [0, 0, 7, 0, 0, 0, 7, 0, 0]
        self.assertNotEquals(labels0, conll2003[0]['ner_tags'])
        selected_dataset_so_far.append((0, {**conll2003[0], 'labels': labels0}))
        selected_dataset_so_far.append((1, {**conll2003[1], 'labels': conll2003[1]['ner_tags']}))

        # We choose what to annotate in the form of 
        # (sentence_id, <list_of_tokens_to_annotate>)
        selections = [(0, [0]), (2, [0])]
        new_dataset = annotate(conll2003, selected_dataset_so_far, selections)

        # We are adding a new sentence. Check that it is there
        self.assertEquals(len(new_dataset), 3)

        new_sentence_with_id0 = [x for x in new_dataset if x[0] == 0][0][1]
        new_sentence_with_id1 = [x for x in new_dataset if x[0] == 1][0][1]
        new_sentence_with_id2 = [x for x in new_dataset if x[0] == 2][0][1]
        
        # Sanity check that we are not doing modifications in-place
        self.assertEquals(selected_dataset_so_far[0][1]['labels'], labels0)

        # sentence 0 had only one token with missing annotations and we 
        # annotated that token. Check that now all labels are annotated
        self.assertEquals(new_sentence_with_id0['labels'], conll2003[0]['ner_tags'])

        # sentence 1 had everything annotated. Check that it still has everything
        # Additionally, sanity checks
        self.assertEquals(new_sentence_with_id1['labels'], conll2003[1]['ner_tags'])
        self.assertEquals(new_sentence_with_id1, selected_dataset_so_far[1][1])

        # sentence 2 was newly added, but only partially. Check that this is still the case
        self.assertNotEquals(new_sentence_with_id2['labels'], conll2003[2]['ner_tags'])
        # Check that the token we annotated is present
        self.assertEquals(new_sentence_with_id2['labels'][0], conll2003[2]['ner_tags'][0])

    """
    Check sentence-level annotation
    """
    def test_annotate_dataset2(self):
        conll2003 = load_dataset("conll2003")['train']

        # The dataset we have selected so far
        # A list of (sentence_id, line_dictionary)
        selected_dataset_so_far = []

        selected_dataset_so_far.append((0, {**conll2003[0]}))
        selected_dataset_so_far.append((1, {**conll2003[1]}))

        # We choose what to annotate in the form of 
        # (sentence_id, <list_of_tokens_to_annotate>)
        selections = [0, 1, 2, 3]
        new_dataset = annotate(conll2003, selected_dataset_so_far, selections)

        # We are adding a new sentence. Check that it is there
        self.assertEquals(len(new_dataset), 4)

        new_sentence_with_id0 = [x for x in new_dataset if x[0] == 0][0][1]
        new_sentence_with_id1 = [x for x in new_dataset if x[0] == 1][0][1]
        new_sentence_with_id2 = [x for x in new_dataset if x[0] == 2][0][1]
        new_sentence_with_id3 = [x for x in new_dataset if x[0] == 3][0][1]

        self.assertEquals(new_sentence_with_id0, selected_dataset_so_far[0][1])
        self.assertEquals(new_sentence_with_id1, selected_dataset_so_far[1][1])
        self.assertEquals(new_sentence_with_id0, conll2003[0])
        self.assertEquals(new_sentence_with_id1, conll2003[1])
        self.assertEquals(new_sentence_with_id2, conll2003[2])
        self.assertEquals(new_sentence_with_id3, conll2003[3])

if __name__ == '__main__':
    unittest.main()




