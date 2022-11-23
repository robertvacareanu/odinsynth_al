import json
from src.arg_parser import get_argparser
from src.dataset_utils import get_conll2003
from src.query_strategies.utils import filter_invalid_token_predictions
from src.utils import ALAnnotation, compute_metrics, init_random, tokenize_and_align_labels, verbose_performance_printing
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification
from transformers import AutoTokenizer
from datasets import Dataset
import evaluate
import random
from collections import Counter
from src.query_strategies.sentence_level_strategies_tc import (
    random_query             as sl_random_query, 
    prediction_entropy_query as sl_prediction_entropy_query, 
    breaking_ties_query      as sl_breaking_ties_query, 
    least_confidence_query   as sl_least_confidence_query
    )
from src.query_strategies.entity_level_strategies_tc import (
    random_query             as el_random_query, 
    prediction_entropy_query as el_prediction_entropy_query, 
    breaking_ties_query      as el_breaking_ties_query, 
    least_confidence_query   as el_least_confidence_query
    )
from src.query_strategies.token_level_strategies_tc import (
    random_query             as tl_random_query, 
    prediction_entropy_query as tl_prediction_entropy_query, 
    breaking_ties_query      as tl_breaking_ties_query, 
    least_confidence_query   as tl_least_confidence_query
    )

"""
TODO Make every query strategy return the following thing:
    List[(Int, List[ALAnnotation])]
So a tuple of integers and list
The first element of the tuple is the sentence id (in the original dataset space)
The second elmenet in the tuple is the ner annotations
This is because we investigate differnet levels of costs:
- sentence level
- entity level
- token level

So we keep such a list in memory
Then, for training, we select using the sentence ids
Then, we iterate over each dict overwriting `labels` using our active_learning_labels
This is done to ignore backpropagation on tokens we don't have annotations for
"""

args = vars(get_argparser().parse_args())
init_random(args['seed'])


annotation_strategy_to_query_strategy_fn = {
    'sentence_level': {
        'random_query'            : sl_random_query,
        'prediction_entropy_query': sl_prediction_entropy_query,
        'breaking_ties_query'     : sl_breaking_ties_query,
        'least_confidence_query'  : sl_least_confidence_query,    
    },
    'entity_level': {
        'random_query'            : el_random_query,
        'prediction_entropy_query': el_prediction_entropy_query,
        'breaking_ties_query'     : el_breaking_ties_query,
        'least_confidence_query'  : el_least_confidence_query,    
    },
    'token_level': {
        'random_query'            : tl_random_query,
        'prediction_entropy_query': tl_prediction_entropy_query,
        'breaking_ties_query'     : tl_breaking_ties_query,
        'least_confidence_query'  : tl_least_confidence_query,    
    },
}


query_strategy_function = annotation_strategy_to_query_strategy_fn[args['annotation_strategy']][args['query_strategy_function']]
query_random = annotation_strategy_to_query_strategy_fn[args['annotation_strategy']]['random_query']

conll2003, label_to_id, id_to_label = get_conll2003()

starting_size = int(len(conll2003['train']) * args['starting_size_ratio'])

selected_indices = random.sample(range(0, len(conll2003['train'])), starting_size)
selected_indices_set = set(selected_indices)
# This list holds what we have selected so far
selected_dataset_so_far = dict([(x, ALAnnotation.from_line(line=conll2003['train'][x], sid=x)) for x in selected_indices])

tokenizer = AutoTokenizer.from_pretrained(args['underlying_model'])

# Tokenize everything
tokenized_conll2003 = conll2003.map(lambda x: tokenize_and_align_labels(tokenizer, x), batched=True)

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, return_tensors="pt")
metric = evaluate.load("seqeval")


number_of_al_iterations = args['number_of_al_iterations']
if len(args['epochs']) < number_of_al_iterations:
    number_of_new_examples_list  = args['number_of_new_examples'] + ([args['number_of_new_examples'][-1]] * (number_of_al_iterations - len(args['number_of_new_examples'])))
else:
    number_of_new_examples_list  = args['number_of_new_examples'][:number_of_al_iterations]
if len(args['epochs']) < number_of_al_iterations:
    epochs_list                  = args['epochs'] + ([args['epochs'][-1]] * (number_of_al_iterations - len(args['epochs'])))
else:
    epochs_list                  = args['epochs'][:number_of_al_iterations]
if len(args['learning_rate']) < number_of_al_iterations:
    learning_rates_list          = args['learning_rate'] + ([args['learning_rate'][-1]] * (number_of_al_iterations - len(args['learning_rate'])))
else:
    learning_rates_list          = args['learning_rate'][:number_of_al_iterations]

all_results = []

for active_learning_iteration, number_of_new_examples, epochs, learning_rate in zip(range(number_of_al_iterations), number_of_new_examples_list, epochs_list, learning_rates_list):
    model = AutoModelForTokenClassification.from_pretrained(args['underlying_model'], num_labels=len(label_to_id))

    # Create the new dataset using all the selected indices
    # Then, overwrite the labels by using only the labels we have annotated so far
    data  = [{**conll2003["train"][x], 'ner_tags': selected_dataset_so_far[x].ner_tags} for x in selected_indices]
    data  = Dataset.from_list(data).map(lambda x: tokenize_and_align_labels(tokenizer, x), batched=True, load_from_cache_file=False)

    # Use `Z` for 'O' to artificialy push it to the end of the sorted list
    selected_data_distribution = sorted(Counter([id_to_label[x] for x in [z for y in selected_dataset_so_far.values() for z in y.get_annotated_tokens()]]).items(), key=lambda x: ('Z', 'Z') if x[0] == 'O' else (x[0][2:], x[0][:2]))
    number_of_annotated_tokens = sum([x[1].number_of_annotated_tokens() for x in list(selected_dataset_so_far.items())])
    print(f"Total number of sentences partially or fully annotated: {len(data)}")
    print(f"Total number of annotated tokens: {sum([x.number_of_annotated_tokens() for x in selected_dataset_so_far.values()])}")
    print(f"Total number of each token type: {selected_data_distribution}")

    training_args = TrainingArguments(
        output_dir="./outputs",
        # evaluation_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        num_train_epochs=epochs,
        weight_decay=0.01,
        save_strategy="no",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=data,
        # eval_dataset=tokenized_conll2003["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda x: compute_metrics(x[0], x[1], id_to_label, metric=metric, verbose=True)
    )

    trainer.train()

    # unlabeled_dataset = []
    # for i in range(len(tokenized_conll2003['train'])):
        # if i not in selected_indices_set:
            # unlabeled_dataset.append(tokenized_conll2003['train'][i])

    predictions = trainer.predict(tokenized_conll2003['train'])

    # Filter the [PAD] scores, the [CLS] scores, etc.
    predictions_without_invalids = filter_invalid_token_predictions(predictions)

    selected_dataset_so_far = dict(
        query_strategy_function(predictions_without_invalids, k=number_of_new_examples, id_to_label=id_to_label, dataset_so_far=list(selected_dataset_so_far.items()), dataset=tokenized_conll2003['train'])
    )
    # selected_dataset_so_far = dict(
        # query_strategy_function(predictions_without_invalids, k=int(3*number_of_new_examples/4), id_to_label=id_to_label, dataset_so_far=list(selected_dataset_so_far.items()), dataset=tokenized_conll2003['train'])
    # )
    # selected_dataset_so_far = dict(
        # query_random(predictions_without_invalids, k=int(1*number_of_new_examples/4), id_to_label=id_to_label, dataset_so_far=list(selected_dataset_so_far.items()), dataset=tokenized_conll2003['train'])
    # )
    selected_indices_set = set(selected_dataset_so_far.keys())
    # We do sorting to avoid any unpredictability that might have been added by the order in the dict 
    # (i.e. set([1,2,3]) is not necessarily guaranteed to have the same order in between successive runs if PYTHONHASHSEED is not set)
    # We let the dataloader do the shuffling during training and we ensure the same order of the dataset initially
    selected_indices = sorted(list(selected_dataset_so_far.keys()))

    predictions_val = trainer.predict(tokenized_conll2003["validation"])
    if args['verbose']:
        verbose_performance_printing(predictions_val, active_learning_iteration)
        
    all_results.append(
        {
            'active_learning_iteration': active_learning_iteration,
            **predictions_val.metrics, 
            # 'all_data_distribution'     : [(id_to_label[x[0]], x[1]) for x in Counter([y for x in conll2003['train'].select(selected_indices)['ner_tags'] for y in x]).items()],
            'annotation_strategy'       : args['annotation_strategy'],
            'query_strategy_function'   : args['query_strategy_function'],
            'number_of_al_iterations'   : args['number_of_al_iterations'],
            'number_of_annotated_tokens': number_of_annotated_tokens,
            'selected_data_distribution': selected_data_distribution,
            'seed'                      : args['seed'],
            'starting_size_ratio'       : args['starting_size_ratio'],
            'underlying_model'          : args['underlying_model'],
            'epochs'                    : epochs,
            'number_of_new_examples'    : number_of_new_examples,
            'learning_rate'             : learning_rate,
        }
    )

if args['append_logs_to_file']:
    with open(args['append_logs_to_file'], 'a+') as fout:
        _=fout.write(json.dumps(all_results))
        _=fout.write('\n')
else:
    print(json.dumps(all_results))

