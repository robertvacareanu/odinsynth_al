"""
Different ways to select data from the initial dataset
There can be different (and better) ways to select the initial
dataset than simply randomly sampling
"""

import random
import numpy as np 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.metrics import pairwise_distances

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

    vectorizer = TfidfVectorizer(stop_words = 'english', ngram_range=(1,2), norm=None)

    X     = vectorizer.fit_transform(train_text)
    means = np.array([top_k_mean(X.getrow(i).data, topk_size) for i in range(X.shape[0])])

    new_starting_size = int(0.8 * starting_size)
    ids   = np.argpartition(means, -starting_size)[-starting_size:]

    return ids.tolist()


def tfidf_kmeans_initial_dataset_sampling(train_text, **kwargs):

    starting_size = kwargs['starting_size']
    vectorizer = TfidfVectorizer(stop_words = 'english', ngram_range=(1,2), norm=None)

    X     = vectorizer.fit_transform(train_text)

    mbkm = MiniBatchKMeans(n_clusters=starting_size, init_size=1024, batch_size=2048, random_state=1).fit(X)

    indices     = []
    indices_set = set()
    for cluster in mbkm.cluster_centers_:
        distances = pairwise_distances(X, cluster.reshape(1, -1), metric='cosine')
        distances = np.delete(distances, indices)
        argmax = distances.argmax()
        indices.append(argmax)
        indices_set.add(argmax)

    return indices



