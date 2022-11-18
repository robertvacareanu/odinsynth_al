import unittest

from src.query_strategies.utils import annotate, take_full_entity
from datasets import load_dataset

from src.utils import ALAnnotation

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
        labels0 = [-100, 0, 7, 0, 0, 0, 7, 0, 0]
        self.assertNotEquals(labels0, conll2003[0]['ner_tags'])
        selected_dataset_so_far.append((0, ALAnnotation.from_line({}, 0, labels0)))
        selected_dataset_so_far.append((1, ALAnnotation.from_line(conll2003[1], 1)))

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
        self.assertEquals(selected_dataset_so_far[0][1].ner_tags, labels0)

        # sentence 0 had only one token with missing annotations and we 
        # annotated that token. Check that now all labels are annotated
        self.assertEquals(new_sentence_with_id0.ner_tags, conll2003[0]['ner_tags'])

        # sentence 1 had everything annotated. Check that it still has everything
        # Additionally, sanity checks
        self.assertEquals(new_sentence_with_id1.ner_tags, conll2003[1]['ner_tags'])
        self.assertEquals(new_sentence_with_id1, selected_dataset_so_far[1][1])

        # sentence 2 was newly added, but only partially. Check that this is still the case
        self.assertNotEquals(new_sentence_with_id2.ner_tags, conll2003[2]['ner_tags'])
        # Check that the token we annotated is present
        self.assertEquals(new_sentence_with_id2.ner_tags[0], conll2003[2]['ner_tags'][0])

    """
    Check sentence-level annotation
    """
    def test_annotate_dataset2(self):
        conll2003 = load_dataset("conll2003")['train']

        # The dataset we have selected so far
        # A list of (sentence_id, line_dictionary)
        selected_dataset_so_far = []

        selected_dataset_so_far.append((0, ALAnnotation.from_line(conll2003[0], 0)))
        selected_dataset_so_far.append((1, ALAnnotation.from_line(conll2003[1], 1)))

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
        self.assertEquals(new_sentence_with_id0.ner_tags, conll2003[0]['ner_tags'])
        self.assertEquals(new_sentence_with_id1.ner_tags, conll2003[1]['ner_tags'])
        self.assertEquals(new_sentence_with_id2.ner_tags, conll2003[2]['ner_tags'])
        self.assertEquals(new_sentence_with_id3.ner_tags, conll2003[3]['ner_tags'])

    def test_take_full_entity(self):
        label_to_id = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7, 'I-MISC': 8}
        id_to_label = {v:k for (k,v) in label_to_id.items()}
        #         0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39
        labels = [0, 1, 2, 2, 2, 1, 2, 1, 2, 3, 4, 4, 4, 0, 1, 2, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0]

        # We give an 'O'
        output = take_full_entity(labels, id_to_label, 0)
        self.assertEqual(len(output), 1)
        self.assertEqual(output, [0])

        # We give an 'O'
        output = take_full_entity(labels, id_to_label, 13)
        self.assertEqual(len(output), 1)
        self.assertEqual(output, [13])

        # We give a B-
        output = take_full_entity(labels, id_to_label, 1)
        self.assertEqual(len(output), 4)
        self.assertEqual(output, [1,2,3,4])

        # We give a B-
        output = take_full_entity(labels, id_to_label, 17)
        self.assertEqual(len(output), 1)
        self.assertEqual(output, [17])

        # We give a B-
        output = take_full_entity(labels, id_to_label, 26)
        self.assertEqual(len(output), 9)
        self.assertEqual(output, [26,27,28,29,30,31,32,33,34])

        # We give a I-
        output = take_full_entity(labels, id_to_label, 2)
        self.assertEqual(len(output), 4)
        self.assertEqual(output, [1,2,3,4])

        # We give a I-
        output = take_full_entity(labels, id_to_label, 8)
        self.assertEqual(len(output), 2)
        self.assertEqual(output, [7, 8])

        # We give a I-
        output = take_full_entity(labels, id_to_label, 27)
        self.assertEqual(len(output), 9)
        self.assertEqual(output, [26,27,28,29,30,31,32,33,34])

        # We give a I-
        output = take_full_entity(labels, id_to_label, 28)
        self.assertEqual(len(output), 9)
        self.assertEqual(output, [26,27,28,29,30,31,32,33,34])

        # We give a I-
        output = take_full_entity(labels, id_to_label, 29)
        self.assertEqual(len(output), 9)
        self.assertEqual(output, [26,27,28,29,30,31,32,33,34])

        # We give a I-
        output = take_full_entity(labels, id_to_label, 30)
        self.assertEqual(len(output), 9)
        self.assertEqual(output, [26,27,28,29,30,31,32,33,34])

        # We give a I-
        output = take_full_entity(labels, id_to_label, 33)
        self.assertEqual(len(output), 9)
        self.assertEqual(output, [26,27,28,29,30,31,32,33,34])

        # We give a I-
        output = take_full_entity(labels, id_to_label, 34)
        self.assertEqual(len(output), 9)
        self.assertEqual(output, [26,27,28,29,30,31,32,33,34])



if __name__ == '__main__':
    unittest.main()




