"""
This file contains various argument checks to ensure (as much as possible) a smooth training run
"""



"""
Side effects. If any check fails, raise a ValueError
"""
def do_arg_checks(args):
    if args['starting_size_ratio'] <= 0 and args['use_ful_dataset'] == False:
        raise ValueError("The starting size ratio must be greater than 0. We cannot train with no examples. Is everything ok?")
    if args['starting_size_ratio'] > 1.0 and args['use_ful_dataset'] == False:
        raise ValueError("The starting size ratio must be smaller or equal with 1. We cannot train with more examples than available in the dataset. Is everything ok?")
    if any([x < 0 for x in args['number_of_new_examples']]) and args['use_ful_dataset'] == False:
        raise ValueError("The number of new examples must be greater than 0. Is everything ok?")
    if any([x < 0 for x in args['epochs']]):
        raise ValueError("The number of epochs must be greater than 0. Is everything ok?")
    if args['train_batch_size'] < 0:
        raise ValueError("The batch size must be greater than 0. Is everything ok?")
    if args['eval_batch_size'] < 0:
        raise ValueError("The batch size must be greater than 0. Is everything ok?")
    if any([x < 0 for x in args['learning_rate']]):
        raise ValueError("The learning rate must be greater than 0. Is everything ok?")
    if any([x > 0.1 for x in args['learning_rate']]):
        raise ValueError("While not exactly wrong, the learning rate seems very big. Is everything ok?")
    if 'fewnerd' in args['dataset_name'] and args['metrics_name'] != 'compute_metrics_fewnerd':
        raise ValueError("FewNerd needs a different evaluation metric because it uses `IO` tag format. Is everything ok?")

 

