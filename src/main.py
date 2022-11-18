import json
from src.arg_parser import get_argparser
from src.dataset_utils import get_conll2003
from src.query_strategies.utils import filter_invalid_token_predictions
from src.utils import compute_metrics, init_random, tokenize_and_align_labels, verbose_performance_printing
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification
from transformers import AutoTokenizer
from datasets import load_dataset
import evaluate
import random
import scipy
from collections import Counter
from src.query_strategies.sentence_level_strategies_tc import random_query, prediction_entropy_query, breaking_ties_query, least_confidence_query

init_random(1)
args = vars(get_argparser().parse_args())

query_strategy = {
    'random_query'            : random_query,
    'prediction_entropy_query': prediction_entropy_query,
    'breaking_ties_query'     : breaking_ties_query,
    'least_confidence_query'  : least_confidence_query,
}
query_strategy_function = query_strategy[args['query_strategy_function']]

conll2003, label_to_id, id_to_label = get_conll2003()

starting_size = int(len(conll2003['train']) * args['starting_size_ratio'])

selected_indices = random.sample(range(0, len(conll2003['train'])), starting_size)
selected_indices_set = set(selected_indices)
selected_dataset_so_far = [conll2003['train'][x]['ner_tags'] for x in selected_indices]

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Tokenize everything
tokenized_conll2003 = conll2003.map(lambda x: tokenize_and_align_labels(tokenizer, x), batched=True)
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, return_tensors="pt")
metric = evaluate.load("seqeval")


number_of_new_examples  = args['number_of_new_examples']
number_of_al_iterations = args['number_of_al_iterations']
epochs                  = [args['epochs']] * number_of_al_iterations
learning_rates          = [args['learning_rate']] * number_of_al_iterations

all_results = []

for active_learning_iteration in range(number_of_al_iterations):
    model = AutoModelForTokenClassification.from_pretrained(args['underlying_model'], num_labels=len(label_to_id))
    data = tokenized_conll2003["train"].select(selected_indices)

    print(f"Size of data: {len(data)}")

    training_args = TrainingArguments(
        output_dir="./outputs",
        # evaluation_strategy="epoch",
        learning_rate=learning_rates[active_learning_iteration],
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        num_train_epochs=epochs[active_learning_iteration],
        weight_decay=0.01,
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

    unlabeled_dataset = []
    for i in range(len(tokenized_conll2003['train'])):
        if i not in selected_indices_set:
            unlabeled_dataset.append(tokenized_conll2003['train'][i])

    predictions = trainer.predict(unlabeled_dataset)

    # Filter the [PAD] scores, the [CLS] scores, etc.
    predictions_without_invalids = filter_invalid_token_predictions(predictions)
    # for (sentence, labels) in zip(predictions.predictions, predictions.label_ids):
    #     current_sentence = []
    #     for (token_scores, label) in zip(sentence, labels):
    #         if label == -100:
    #             continue
    #         else:
    #             current_sentence.append(scipy.special.softmax(token_scores, axis=0).tolist())

        # predictions_without_invalids.append(current_sentence)

    new_indices = query_strategy_function(predictions_without_invalids, k=number_of_new_examples)
    selected_indices_set = selected_indices_set.union(new_indices)
    selected_indices += new_indices

    predictions_val = trainer.predict(tokenized_conll2003["validation"])
    verbose_performance_printing(predictions_val, active_learning_iteration)
    all_results.append({'active_learning_iteration': active_learning_iteration, **predictions_val.metrics, 'data_distribution': [(id_to_label[x[0]], x[1]) for x in Counter([y for x in conll2003['train'].select(selected_indices)['ner_tags'] for y in x]).items()], 'query_strategy_function': args['query_strategy_function'], 'number_of_new_examples': args['number_of_new_examples'], 'number_of_al_iterations': args['number_of_al_iterations'],})

print(json.dumps(all_results))
