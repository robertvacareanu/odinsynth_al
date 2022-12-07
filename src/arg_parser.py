"""
This file contains the necessary code for command-line arguments
Overall, we support things like:
- learning rate
- number of epochs

Particular to AL, we support
- number of active learning iterations
- number of new examples per iteration
- different learning rates and different epochs for each active learning iteration
- various query strategies
- various annotation strategies

The same code can be used to train the model in a fully supervised way, by using the `use_full_dataset` flag

"""

import argparse

def base_parser(parent_parser):
    subparser = parent_parser.add_argument_group("active_learning")
    subparser.add_argument('--seed', type=int, default=1)
    subparser.add_argument('--starting_size_ratio', type=float, default = 0.01)
    subparser.add_argument('--underlying_model', type=str, default="distilbert-base-uncased")
    subparser.add_argument('--number_of_new_examples', type=int, nargs='+', default=[20], help="Number of new examples. We allow for multiple values here, one for each active learning iteration. If not enoguh, we duplicate the last one")
    subparser.add_argument('--number_of_al_iterations', type=int, default=10)
    subparser.add_argument('--epochs', type=int, nargs='+', default=[10], help="Number of epochs to train in each active learning iteration. We allow for multiple epoch settings, one for each active learning iteration. If not enough, we duplicate the last one")
    subparser.add_argument('--learning_rate', type=float, nargs='+', default=[2e-5], help="Learning rate. We allow for multiple learning rates, one for each active learning iteration. If not enough, we duplicate the last one")
    subparser.add_argument('--query_strategy_function', type=str, default='random_query', choices=['random_query', 'prediction_entropy_query', 'breaking_ties_query', 'least_confidence_query'])
    subparser.add_argument('--annotation_strategy', type=str, default='sentence_level', choices=['sentence_level', 'entity_level', 'token_level'])
    subparser.add_argument('--append_logs_to_file', type=str, default=None)
    subparser.add_argument('--verbose', action='store_true')
    subparser.add_argument('--fp16', action='store_true')
    subparser.add_argument('--dataset_name', type=str, default='conll2003', choices=['conll2003', 'ontonotes', 'fewnerd_cg', 'fewnerd_fg'])
    subparser.add_argument('--early_stopping_patience', type=int, default=3)
    subparser.add_argument('--train_batch_size', type=int, default=8)
    subparser.add_argument('--eval_batch_size', type=int, default=16)
    subparser.add_argument('--initial_dataset_selection_strategy_top_k', type=int, default=5)
    subparser.add_argument('--use_full_dataset', action='store_true', help='If set, we will use the full dataset and would not perform any active learning')
    subparser.add_argument('--training_annotation_style', type=str, default='mask_all_unknown', choices=['mask_all_unknown', 'drop_all_unknown', 'mask_entity_looking_unknowns', 'drop_entity_looking_unknowns', 'dynamic_window'])
    subparser.add_argument('--initial_dataset_selection_strategy', type=str, default='random_initial_dataset_sampling', choices=['random_initial_dataset_sampling', 'tfidf_initial_dataset_sampling', 'longest_sentences_dataset_sampling', 'tfidf_kmeans_initial_dataset_sampling'])
    subparser.add_argument('--use_postags_for_selection', action='store_true', help="If set, we will do the sampling using the part-of-speech tags, in case the `initial_dataset_selection_strategy` can use them.")
    return parent_parser


def get_argparser():
    parser = argparse.ArgumentParser(description='AL', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    base_parser(parser)
    return parser
