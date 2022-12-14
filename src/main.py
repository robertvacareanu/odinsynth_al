import glob
import json
from collections import Counter
import math
import shutil
import tqdm

from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification
from transformers import AutoTokenizer
from transformers.trainer_callback import EarlyStoppingCallback
from datasets import Dataset
import evaluate

from src.arg_checks import do_arg_checks
from src.arg_parser import get_argparser
from src.dataset_utils import get_conll2003, get_fewnerd_cg, get_fewnerd_fg, get_ontonotes
from src.query_strategies.utils import filter_invalid_token_predictions
from src.utils import ALAnnotation, compute_metrics, init_random, tokenize_and_align_labels, verbose_performance_printing
from src.dataset_selection_strategies.strategies import (
    longest_sentences_dataset_sampling, 
    random_initial_dataset_sampling, 
    tfidf_initial_dataset_sampling, 
    tfidf_probabilistic_initial_dataset_sampling, 
    tfidf_kmeans_initial_dataset_sampling,
    tfidf_most_dissimilar_initial_dataset_sampling,
    supervised_initial_dataset_sampling,
    tfidf_avoiding_duplicates_initial_dataset_sampling,
    supervised_avoid_duplicates_initial_dataset_sampling
    )

from src.query_strategies.sentence_level_strategies_tc import (
    random_query                   as sl_random_query, 
    prediction_entropy_query       as sl_prediction_entropy_query, 
    breaking_ties_query            as sl_breaking_ties_query, 
    breaking_ties_bernoulli_query  as sl_breaking_ties_bernoulli_query, 
    least_confidence_query         as sl_least_confidence_query
    )
from src.query_strategies.entity_level_strategies_tc import (
    random_query                   as el_random_query, 
    prediction_entropy_query       as el_prediction_entropy_query, 
    breaking_ties_query            as el_breaking_ties_query, 
    breaking_ties_bernoulli_query  as el_breaking_ties_bernoulli_query, 
    least_confidence_query         as el_least_confidence_query
    )
from src.query_strategies.token_level_strategies_tc import (
    random_query                   as tl_random_query, 
    prediction_entropy_query       as tl_prediction_entropy_query, 
    breaking_ties_query            as tl_breaking_ties_query, 
    breaking_ties_bernoulli_query  as tl_breaking_ties_bernoulli_query, 
    least_confidence_query         as tl_least_confidence_query
    )

"""
Every query strategy return the following thing:
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
do_arg_checks(args)
print(args)

init_random(args['seed'])


annotation_strategy_to_query_strategy_fn = {
    'sentence_level': {
        'random_query'                  : sl_random_query,
        'prediction_entropy_query'      : sl_prediction_entropy_query,
        'breaking_ties_query'           : sl_breaking_ties_query,
        'breaking_ties_bernoulli_query' : sl_breaking_ties_bernoulli_query,
        'least_confidence_query'        : sl_least_confidence_query,    
    },
    'entity_level': {
        'random_query'                  : el_random_query,
        'prediction_entropy_query'      : el_prediction_entropy_query,
        'breaking_ties_query'           : el_breaking_ties_query,
        'breaking_ties_bernoulli_query' : el_breaking_ties_bernoulli_query,
        'least_confidence_query'        : el_least_confidence_query,    
    },
    'token_level': {
        'random_query'                  : tl_random_query,
        'prediction_entropy_query'      : tl_prediction_entropy_query,
        'breaking_ties_query'           : tl_breaking_ties_query,
        'breaking_ties_bernoulli_query' : tl_breaking_ties_bernoulli_query,
        'least_confidence_query'        : tl_least_confidence_query,    
    },
}

initial_dataset_sampling_to_fn = {
    'random_initial_dataset_sampling'                     : random_initial_dataset_sampling,
    'tfidf_initial_dataset_sampling'                      : tfidf_initial_dataset_sampling,
    'longest_sentences_dataset_sampling'                  : longest_sentences_dataset_sampling,
    'tfidf_probabilistic_initial_dataset_sampling'        : tfidf_probabilistic_initial_dataset_sampling,
    'tfidf_kmeans_initial_dataset_sampling'               : tfidf_kmeans_initial_dataset_sampling,
    'tfidf_most_dissimilar_initial_dataset_sampling'      : tfidf_most_dissimilar_initial_dataset_sampling,
    'supervised_initial_dataset_sampling'                 : supervised_initial_dataset_sampling,
    'tfidf_avoiding_duplicates_initial_dataset_sampling'  : tfidf_avoiding_duplicates_initial_dataset_sampling,
    'supervised_avoid_duplicates_initial_dataset_sampling': supervised_avoid_duplicates_initial_dataset_sampling,
}

dataset_name_to_fn = {
    'conll2003' : get_conll2003,
    'ontonotes' : get_ontonotes,
    'fewnerd_cg': get_fewnerd_cg,
    'fewnerd_fg': get_fewnerd_fg,
}


query_strategy_function = annotation_strategy_to_query_strategy_fn[args['annotation_strategy']][args['query_strategy_function']]
query_random = annotation_strategy_to_query_strategy_fn[args['annotation_strategy']]['random_query']

dataset_name=args['dataset_name']
ner_dataset, label_to_id, id_to_label = dataset_name_to_fn[dataset_name](args)

total_number_of_tokens_available = len([y for x in ner_dataset['train'] for y in x['tokens']])

if args['use_full_dataset']:
    starting_size_ratio = 1.0
    starting_size       = len(ner_dataset['train'])
else:
    starting_size_ratio = args['starting_size_ratio']
    starting_size       = min(int(len(ner_dataset['train']) * starting_size_ratio), len(ner_dataset['train']))

# selected_indices = initial_dataset_sampling_to_fn[args['initial_dataset_selection_strategy']]([' '.join(x) for x in ner_dataset['train']['tokens']], starting_size=starting_size, top_k_size=args['initial_dataset_selection_strategy_top_k'])
if not args['use_full_dataset']:
    text     = []
    ner_tags = []
    
    for line in ner_dataset['train']:
        sent = []
        ners = []
        if args['use_postags_for_selection']:
            for token, tag, ner in zip(line['tokens'], line['pos_tags_text'], line['ner_tags']):
                if 'NNP' in tag:
                    sent.append(token)
                    ners.append(id_to_label[ner])
        else:
            for token, ner in zip(line['tokens'], line['ner_tags']):
                sent.append(token)
                ners.append(id_to_label[ner])
        text.append(' '.join(sent))
        ner_tags.append(' '.join(ners))
    selected_indices = initial_dataset_sampling_to_fn[args['initial_dataset_selection_strategy']](text, starting_size=starting_size, top_k_size=args['initial_dataset_selection_strategy_top_k'], ner_tags=ner_tags, params={'stop_words': args['stop_words'], 'ngram_range1': args['ngram_range1'], 'ngram_range2': args['ngram_range2']})
else:
    selected_indices = list(range(len(ner_dataset['train'])))
print("Selected indices: ", selected_indices)
selected_indices_set = set(selected_indices)

# This list holds what we have selected so far
selected_dataset_so_far = dict([(x, ALAnnotation.from_line(line=ner_dataset['train'][x], sid=x)) for x in selected_indices])

tokenizer = AutoTokenizer.from_pretrained(args['underlying_model'])

print("Tokenized everything")
# Tokenize everything
tokenized_ner_dataset = ner_dataset.map(lambda x: tokenize_and_align_labels(tokenizer, x), batched=True)
print("Everything tokenized")

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
if len(args['early_stopping_patience']) < number_of_al_iterations:
    early_stopping_patience_list = args['early_stopping_patience'] + ([args['early_stopping_patience'][-1]] * (number_of_al_iterations - len(args['early_stopping_patience'])))
else:
    early_stopping_patience_list = args['early_stopping_patience'][:number_of_al_iterations]

all_results = []

for active_learning_iteration, number_of_new_examples, epochs, learning_rate, early_stopping_patience in zip(range(number_of_al_iterations), number_of_new_examples_list, epochs_list, learning_rates_list, early_stopping_patience_list):
    # Remove all checkpoints before starting a new training procedure
    for f in glob.glob(f'./outputs/{dataset_name}/checkpoint-*'):
        shutil.rmtree(f)
    model = AutoModelForTokenClassification.from_pretrained(args['underlying_model'], num_labels=len(label_to_id))

    # Create the new dataset using all the selected indices
    # Then, overwrite the labels by using only the labels we have annotated so far
    data  = []
    # print("Prepare the dataset")
    for x in tqdm.tqdm(selected_indices):
        for sdsf in selected_dataset_so_far[x].get_training_annotations(args['training_annotation_style']):
            data.append(sdsf)

    data  = Dataset.from_list(data).map(lambda x: tokenize_and_align_labels(tokenizer, x), batched=True, load_from_cache_file=False)
    print("Dataset tokenized")

    # Use `Z` for 'O' to artificialy push it to the end of the sorted list
    selected_data_distribution = sorted(Counter([id_to_label[x] for x in [z for y in selected_dataset_so_far.values() for z in y.get_annotated_tokens()]]).items(), key=lambda x: ('Z', 'Z') if x[0] == 'O' else (x[0][2:], x[0][:2]))
    number_of_annotated_tokens = sum([x[1].number_of_annotated_tokens() for x in list(selected_dataset_so_far.items())])
    total_number_of_unmasked_tokens = len([y for x in data['labels'] for y in x if y != -100])
    number_of_annotated_nonO_tokens = sum([1 for x in selected_dataset_so_far.values() for y in x.get_annotated_tokens() if y != 0])
    print(f"Total number of selected indices: {len(selected_indices)}")
    print(f"Total number of training sentences partially or fully annotated: {len(data)}")
    print(f"Total number of annotated tokens: {sum([x.number_of_annotated_tokens() for x in selected_dataset_so_far.values()])}")
    print(f"Total number of non-O tokens: {number_of_annotated_nonO_tokens}")
    print(f"Total number of each token type: {selected_data_distribution}")
    print(f"Total number of unmasked tokens: {total_number_of_unmasked_tokens}")
    print(f"Total number of tokens available in the dataset: {total_number_of_tokens_available}")
    print(f"Total percentage of annotated tokens: {(number_of_annotated_tokens/total_number_of_tokens_available) * 100}")

    training_args = TrainingArguments(
        output_dir=f"./outputs/{dataset_name}",
        evaluation_strategy="steps",
        save_steps=math.ceil(len(data)/(args['train_batch_size'] * 2)), # Twice every epoch
        eval_steps=math.ceil(len(data)/(args['train_batch_size'] * 2)), # Twice every epoch
        learning_rate=learning_rate,
        per_device_train_batch_size=args['train_batch_size'],
        per_device_eval_batch_size=args['eval_batch_size'],
        num_train_epochs=epochs,
        weight_decay=0.01,
        save_strategy="steps",
        load_best_model_at_end=True, 
        metric_for_best_model='overall_f1', 
        greater_is_better=True,
        overwrite_output_dir=True,
        fp16=args['fp16']
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=data,
        eval_dataset=tokenized_ner_dataset["val_train"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda x: compute_metrics(x[0], x[1], id_to_label, metric=metric, verbose=True),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)],
    )

    trainer.train()

    # unlabeled_dataset = []
    # for i in range(len(tokenized_ner_dataset['train'])):
        # if i not in selected_indices_set:
            # unlabeled_dataset.append(tokenized_ner_dataset['train'][i])

    # If we do not use the full dataset, then we apply the query strategy function
    if not args['use_full_dataset']:
        predictions = trainer.predict(tokenized_ner_dataset['train'])

        # Filter the [PAD] scores, the [CLS] scores, etc.
        predictions_without_invalids = filter_invalid_token_predictions(predictions)

        # Overwrite the selected dataset so far
        # Particular to our implementation, the new 
        # data will be added to the existing data in 
        # the query strategy function
        selected_dataset_so_far = dict(
            query_strategy_function(predictions_without_invalids, k=number_of_new_examples, id_to_label=id_to_label, dataset_so_far=list(selected_dataset_so_far.items()), dataset=tokenized_ner_dataset['train'])
        )

        selected_indices_set = set(selected_dataset_so_far.keys())
        
        # We do sorting to avoid any unpredictability that might have been added by the order in the dict 
        # (i.e. set([1,2,3]) is not necessarily guaranteed to have the same order in between successive runs if PYTHONHASHSEED is not set)
        # We let the dataloader do the shuffling during training and we ensure the same order of the dataset initially
        selected_indices = sorted(list(selected_dataset_so_far.keys()))

    predictions_val  = trainer.predict(tokenized_ner_dataset["validation"])
    predictions_tval = trainer.predict(tokenized_ner_dataset["val_train"])
    predictions_test = trainer.predict(tokenized_ner_dataset["test"])
    if args['verbose']:
        # print("----------------------")
        # print("------VALIDATION------")
        verbose_performance_printing(predictions_val, active_learning_iteration)
        # print("------VALIDATION------")
        # print("----------------------")
        # print("----------------------")
        # print("---TRAIN-VALIDATION---")
        # verbose_performance_printing(predictions_tval, active_learning_iteration)
        # print("---TRAIN-VALIDATION---")
        # print("----------------------")
        # print("----------------------")
        # print("----------------------")
        # print("---------TEST---------")
        # # verbose_performance_printing(predictions_test, active_learning_iteration)
        # print("---------TEST---------")
        # print("----------------------")
        
    all_results.append(
        {
            'active_learning_iteration': active_learning_iteration,
            'val_metrics'                       : {**predictions_val.metrics}, 
            'train-val_metrics'                 : {**predictions_tval.metrics}, 
            'test_metrics'                      : {**predictions_test.metrics}, # We record test metrics for efficiency purposes, to avoid re-running everything to gather them; Decisions are made on validation only
            # 'all_data_distribution'     : [(id_to_label[x[0]], x[1]) for x in Counter([y for x in ner_dataset['train'].select(selected_indices)['ner_tags'] for y in x]).items()],
            'annotation_strategy'               : args['annotation_strategy'],
            'query_strategy_function'           : args['query_strategy_function'],
            'number_of_al_iterations'           : args['number_of_al_iterations'],
            'number_of_annotated_tokens'        : number_of_annotated_tokens,
            'number_of_annotated_nonO_tokens'   : number_of_annotated_nonO_tokens,
            'total_number_of_unmasked_tokens'   : total_number_of_unmasked_tokens,
            'percentage_of_annotated_tokens'    : (number_of_annotated_tokens/total_number_of_tokens_available) * 100,
            'selected_data_distribution'        : selected_data_distribution,
            'seed'                              : args['seed'],
            'starting_size_ratio'               : starting_size_ratio,
            'underlying_model'                  : args['underlying_model'],
            'epochs'                            : epochs,
            'number_of_new_examples'            : number_of_new_examples,
            'learning_rate'                     : learning_rate,
            'initial_dataset_selection_strategy': args['initial_dataset_selection_strategy'],
            'underlying_model'                  : args['underlying_model'],
            'training_annotation_style'         : args['training_annotation_style'],
            'dataset_name'                      : dataset_name,
            'total_number_of_tokens_available'  : total_number_of_tokens_available,
            'use_postags_for_selection'         : args['use_postags_for_selection'],
            'all_args'                          : json.dumps(args),
        }
    )

if args['append_logs_to_file']:
    with open(args['append_logs_to_file'], 'a+') as fout:
        _=fout.write(json.dumps(all_results))
        _=fout.write('\n')
else:
    print(json.dumps(all_results))

