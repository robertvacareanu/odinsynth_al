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
    subparser.add_argument('--dataset_name', type=str, default='conll2003', choices=['conll2003', 'ontonotes'])
    subparser.add_argument('--train_batch_size', type=int, default=8)
    subparser.add_argument('--eval_batch_size', type=int, default=16)
    subparser.add_argument('--use_full_dataset', action='store_true', help='If set, we will use the full dataset and would not perform any active learning')
    return parent_parser


def get_argparser():
    parser = argparse.ArgumentParser(description='AL', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    base_parser(parser)
    return parser
