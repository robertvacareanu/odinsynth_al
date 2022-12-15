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

    params = kwargs.get('params', {})
    starting_size = kwargs['starting_size']
    topk_size     = kwargs.get('top_k_size', 5)

    vectorizer = TfidfVectorizer(stop_words=params.get('stop_words', 'english'), ngram_range=(params.get('ngram_range1', 1),params.get('ngram_range2', 2)), norm=None)

    X     = vectorizer.fit_transform(train_text)
    means = np.array([top_k_mean(X.getrow(i).data, topk_size) for i in range(X.shape[0])])

    new_starting_size = int(0.8 * starting_size)
    ids   = np.argpartition(means, -starting_size)[-starting_size:]

    return ids.tolist()

def tfidf_probabilistic_initial_dataset_sampling(train_text, **kwargs):

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
    means = np.array([top_k_mean(X.getrow(i).data, topk_size) for i in range(X.shape[0])]) / 5 # ** 2

    # from numpy import savetxt

    # savetxt('z_means.csv', means, delimiter=';')
    import scipy
    probs = scipy.special.softmax(means)

    new_starting_size = int(0.8 * starting_size)
    ids   = np.random.choice(np.arange(len(train_text)), size=starting_size, p=probs, replace=False)

    return ids.tolist()


def tfidf_kmeans_initial_dataset_sampling(train_text, **kwargs):

    starting_size = kwargs['starting_size']
    vectorizer = TfidfVectorizer(stop_words = 'english', ngram_range=(1,1), norm='l2')

    X     = vectorizer.fit_transform(train_text)

    mbkm = MiniBatchKMeans(n_clusters=starting_size, init_size=1024, batch_size=2048, random_state=1).fit(X)

    indices     = []
    for cluster in mbkm.cluster_centers_:
        distances = pairwise_distances(X, cluster.reshape(1, -1), metric='cosine')
        distances_original_idx = np.arange(distances.shape[0]).reshape(-1, 1)

        # Delete already selected indices, as we cannot select them anymore
        distances = np.delete(distances, indices)
        distances_original_idx = np.delete(distances_original_idx, indices)

        # Break ties randomly
        argmax = distances_original_idx[np.random.choice(np.flatnonzero(distances == distances.max()))]

        indices.append(argmax.item())

    return indices


def tfidf_most_dissimilar_initial_dataset_sampling(train_text, **kwargs):

    starting_size = kwargs['starting_size']
    vectorizer = TfidfVectorizer(stop_words = 'english', ngram_range=(1,1), norm='l2')

    X     = vectorizer.fit_transform(train_text)
    distances = pairwise_distances(X, X, metric='cosine')
    distances_sum = distances.min(axis=1)
    ind  = np.argpartition(distances_sum, -starting_size)[-starting_size:]

    return ind.tolist()

"""
Look at the ner tags before selecting
This would represent something akin to an 
upperbound, since in practice we do not possess
such information
"""
def supervised_initial_dataset_sampling(train_text, **kwargs):
    starting_size  = kwargs['starting_size']

    ner_names_text      = [x.split() for x in kwargs.get('ner_tags')]
    nonO_ner_names_text = [[y for y in x if y != 'O'] for x in ner_names_text]
    tokens_text         = [x.split() for x in train_text]

    # We want strings to be able to handle BIO tags
    assert(isinstance(ner_names_text[0][0], str))

    unique_tags = list(set([y[2:] if ('B-' in y or 'I-' in y) else y for x in ner_names_text for y in x]))
    # We don't consider `O`. Almost every sentence will have some `O`s
    unique_tags = [x for x in unique_tags if x != 'O']
    print(unique_tags)

    indices     = []
    indices_set = set()

    for t in unique_tags:
        new_ners = [(i, len([y for y in x if t in y])) for (i, x) in enumerate(ner_names_text)]
        new_ners = sorted(new_ners, key=lambda x: -x[1])
        i = 0
        j = 0
        while i < starting_size//len(unique_tags):
            if new_ners[j][0] not in indices_set:
                indices.append(new_ners[j][0])
                indices_set.add(new_ners[j][0])
                i += 1
            j += 1


    i = 0
    j = 0
    sorted_nonO_ner_names_text = sorted(list(enumerate([len(x) for x in nonO_ner_names_text])), key=lambda x: -x[1])
    while i <= starting_size - len(indices):
        if sorted_nonO_ner_names_text[j][0] not in indices_set:
            indices.append(sorted_nonO_ner_names_text[j][0])
            indices_set.add(sorted_nonO_ner_names_text[j][0])
            i += 1
        j += 1

    return indices


def tfidf_avoiding_duplicates_initial_dataset_sampling(train_text, **kwargs):

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

    vectorizer = TfidfVectorizer(stop_words = 'english', ngram_range=(1,1), norm=None)

    X     = vectorizer.fit_transform(train_text)

    new_starting_size = int(0.8 * starting_size)

    selected_indices     = []
    selected_indices_set = set()
    import tqdm
    for _ in tqdm.tqdm(range(starting_size)):
        means = sorted(enumerate([top_k_mean(X.getrow(i).data, topk_size) for i in range(X.shape[0])]), key=lambda x: -x[1])

        i = 0
        (sid, mean) = means[i]

        while sid in selected_indices_set:
            i += 1
            (sid, mean) = means[i]

        # Decrease the weight of all the sentences containing same words
        for si in X.getrow(sid).indices:
            for row_id in range(X.shape[0]):
                if X[row_id, si] != 0:
                    # print("############")
                    # print(X[row_id, si])
                    X[row_id, si] = X[row_id, si] / 2
                    # print(X[row_id, si])
                    # print("############")
                    # print("\n")

        selected_indices.append(sid)
        selected_indices_set.add(sid)
            
    return selected_indices


"""
Look at the ner tags before selecting
This would represent something akin to an 
upperbound, since in practice we do not possess
such information
"""
def supervised_avoid_duplicates_initial_dataset_sampling(train_text, **kwargs):
    starting_size  = kwargs['starting_size']

    ner_names_text      = [x.split() for x in kwargs.get('ner_tags')]

    # We want strings to be able to handle BIO tags
    assert(isinstance(ner_names_text[0][0], str))

    
    nonO_ner_names_text = [[y for y in x if y != 'O'] for x in ner_names_text]
    tokens_text         = [x.split() for x in train_text]

    tokens_ner_text     = [(i, [to for (to, ta) in zip(x, y) if ta != 'O']) for (i, (x, y)) in enumerate(zip(tokens_text, ner_names_text))]

    selected_ner_names = set()

    selected_indices     = []
    selected_indices_set = set()

    
    import tqdm
    for _ in tqdm.tqdm(range(starting_size)):
        most_ners = sorted(tokens_ner_text, key=lambda x: (-len([y for y in x[1] if y not in selected_ner_names]), random.randint(0, len(train_text)-1)))

        i = 0
        (sid, tokens) = most_ners[i]

        while sid in selected_indices_set:
            i += 1
            (sid, tokens) = most_ners[i]

        for tok in tokens:
            selected_ner_names.add(tok)
        
        selected_indices.append(sid)
        selected_indices_set.add(sid)

    
    return selected_indices














