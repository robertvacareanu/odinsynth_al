import argparse

def base_parser(parent_parser):
    subparser = parent_parser.add_argument_group("active_learning")
    subparser.add_argument('--starting_size_ratio', type=float, default = 0.01)
    subparser.add_argument('--underlying_model', type=str, default="distilbert-base-uncased")
    subparser.add_argument('--number_of_new_examples', type=int, default=20)
    subparser.add_argument('--number_of_al_iterations', type=str, default="distilbert-base-uncased")
    subparser.add_argument('--epochs', type=int, default=10)
    subparser.add_argument('--learning_rate', type=float, default=2e-5)

    return parent_parser


def get_argparser():
    parser = argparse.ArgumentParser(description='AL', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    base_parser(parser)
    return parser