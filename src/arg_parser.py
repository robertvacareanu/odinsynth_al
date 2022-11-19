import argparse

def base_parser(parent_parser):
    subparser = parent_parser.add_argument_group("active_learning")
    subparser.add_argument('--starting_size_ratio', type=float, default = 0.01)
    subparser.add_argument('--underlying_model', type=str, default="distilbert-base-uncased")
    subparser.add_argument('--number_of_new_examples', type=int, default=20)
    subparser.add_argument('--number_of_al_iterations', type=int, default=10)
    subparser.add_argument('--epochs', type=int, default=10)
    subparser.add_argument('--learning_rate', type=float, default=2e-5)
    subparser.add_argument('--query_strategy_function', type=str, default='random_query', choices=['random_query', 'prediction_entropy_query', 'breaking_ties_query', 'least_confidence_query'])
    subparser.add_argument('--annotation_strategy', type=str, default='sentence_level', choices=['sentence_level', 'entity_level', 'token_level'])
    return parent_parser


def get_argparser():
    parser = argparse.ArgumentParser(description='AL', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    base_parser(parser)
    return parser
