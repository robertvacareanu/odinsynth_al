"""
Different ways to select data from the initial dataset
There can be different (and better) ways to select the initial
dataset than simply randomly sampling
"""

import random
import numpy as np 
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text

def random_initial_dataset_sampling(train_text, **kwargs):
    starting_size    = kwargs['starting_size']
    selected_indices = random.sample(range(0, len(train_text)), starting_size)
    return selected_indices

def longest_sentences_dataset_sampling(train_text, **kwargs):
    starting_size       = kwargs['starting_size']
    train_text_with_idx = list(enumerate(train_text))
    sorted_data = sorted(train_text_with_idx, key=lambda x: -len(x[1].split(' ')))
    return [x[0] for x in sorted_data[:starting_size]]

def tfidf_initial_dataset_sampling(train_text, **kwargs):

    def top_k_mean(array, k):
        if np.sum(array) < 1e-6:
            return 0.0
        # return np.max(array)
        if k >= len(array):
            return array.mean()
        ind  = np.argpartition(array, -k)[-k:]
        topk = array[ind]
        return topk.mean()

    starting_size = kwargs['starting_size']
    topk_size     = kwargs.get('top_k_size', 5)

    # stopwords = text.ENGLISH_STOP_WORDS

    # original_index_map = [(i, x) for (i, x) in enumerate(train_text)]
    # filtered_index_map = [x for x in original_index_map if any(y not in stopwords for y in x[1].split(' '))]
    # filtered_index_to_original_index = dict([(i, x[0]) for (i, x) in enumerate(filtered_index_map)])
    # filtered_text = [x[1] for x in filtered_index_map]
    # print(filtered_index_map)

    vectorizer = TfidfVectorizer(stop_words = 'english', ngram_range=(1,2), norm=None)
    X = vectorizer.fit_transform(train_text)
    # means = []
    # for i in range(X.shape[0]):
    #     # print(train_text[i])
    #     # print(X.getrow(i).data)
    #     # print(top_k_mean(X.getrow(i).data, topk_size))
    #     # exit()
    #     m = top_k_mean(X.getrow(i).data, topk_size)
    #     means.append(m)
    # means = np.array(means)
    means = np.array([top_k_mean(X.getrow(i).data, topk_size) for i in range(X.shape[0])])

    new_starting_size = int(0.8 * starting_size)
    ids   = np.argpartition(means, -starting_size)[-starting_size:]
    # ids   = [filtered_index_to_original_index[x] for x in ids]
    # exit()
    return ids.tolist()


