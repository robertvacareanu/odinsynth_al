"""
Different ways to select data from the initial dataset
There can be different (and better) ways to select the initial
dataset than simply randomly sampling
"""

from collections import defaultdict
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


"""
Mostly manually-guided selection
"""
def static_initial_dataset_sampling(train_text, **kwargs):

    # return selected_indices
    # return [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 20, 21, 22, 24, 25, 26, 27, 29, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 53, 54, 56, 57, 58, 59, 60, 61, 62, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 77, 78, 79, 80, 81, 82, 83, 84, 86, 87, 88, 89, 92, 93, 94, 96, 97, 98, 99, 101, 102, 103, 104, 105, 107, 108, 109, 110, 111, 113, 115, 116, 117, 118, 120, 121, 122, 123, 124, 127, 128, 129, 130, 131, 132, 133, 135, 136, 138, 139, 142, 143, 144, 145, 146, 147, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 166, 167, 168]
    # return [5644, 11164, 1692, 7465, 9557, 11800, 5160, 3662, 2392, 8491, 11451, 2420, 3257, 1858, 12859, 12006, 1572, 2778, 958, 3738, 5976, 3279, 8606, 4553, 721, 6657, 9527, 2404, 11160, 5391, 5742, 7210, 1700, 8206, 10187, 8245, 1334, 6068, 8098, 3966, 4701, 108, 13535, 13495, 12789, 2913, 272, 10293, 460, 6140, 11539, 3656, 2032, 475, 5695, 10013, 3942, 3967, 6435, 1324, 7504, 7012, 8271, 8381, 10353, 13809, 5279, 11143, 5820, 12755, 10872, 2092, 5830, 8836, 3935, 5018, 2348, 11704, 3592, 12877, 7866, 13232, 1296, 9771, 10890, 8780, 10949, 4547, 9736, 9707, 3139, 2757, 6076, 12967, 13227, 12921, 8273, 11990, 6231, 3027, 13858, 3791, 5033, 9855, 3154, 4762, 4733, 7123, 9986, 9694, 4840, 3692, 12906, 9751, 1191, 10619, 8179, 6663, 1644, 6628, 7920, 1844, 6789, 3848, 10384, 10794, 4047, 52, 11071, 4891, 4658, 8309, 7081, 9946, 7512, 8736, 1211, 6291, 10581]
    return [5391, 7504, 5830, 5018, 13232, 8780, 9707, 8273, 6231, 8179, 6628, 7512, 7469, 10992, 11277, 7021, 2051, 8312, 7009, 12315, 3148, 7880, 7646, 13507, 7480, 12605, 9649, 13768, 12371, 4535, 5025, 12397, 9940, 1224, 3744, 5075, 13014, 3650, 6447, 11123, 13566, 6785, 7972, 4016, 11218, 13702, 8984, 13283, 417, 7603, 12635, 2874, 4005, 5547, 7319, 13725, 10934, 13166, 8621, 4003, 10086, 670, 12834, 7947, 494, 3475, 6145, 13093, 11788, 7025, 12525, 3746, 11600, 2977, 3272, 6205, 4110, 636, 13090, 9856, 3359, 3309, 13599, 12931, 4633, 11029, 7206, 5877, 2082, 12435, 13394, 10426, 3777, 3578, 8914, 3739, 7404, 12913, 10012, 5731, 13534, 10526, 5938, 9916, 584, 11443, 13185, 13423, 2011, 13353, 12857, 11188, 228, 407, 6606, 10015, 4291, 8423, 525, 11792, 2058, 5092, 7667, 469, 5800, 4561, 13101, 1797, 5165, 3841, 3217, 1645, 8177, 5836, 11900, 2625, 178, 261, 9508]


"""
Look at how often the particular token is an nnp
Favor sentences with more NNPs
"""
def nnp_frequency_initial_dataset_sampling(train_text, **kwargs):
    starting_size  = kwargs['starting_size']

    pos_tags = [x.split() for x in kwargs.get('pos_tags')]
    text     = [x.split() for x in train_text]

    token_to_nnp_count = defaultdict(int)
    token_count        = defaultdict(int)
    for sent, pos_tag in zip(text, pos_tags):
        for token, tag in zip(sent, pos_tag):
            token_count[token] += 1
            if tag == 'NNP':
                token_to_nnp_count[token] += 1

    scores = []
    for sent, pos_tag in zip(text, pos_tags):
        if sum([1 if 'NNP' in x else 0 for x in pos_tag]) > 0:
            score = sum([token_to_nnp_count[x]/token_count[x] for x, t in zip(sent, pos_tag) if 'NNP' in t])/sum([1 if 'NNP' in x else 0 for x in pos_tag])
        else:
            score = 0.0
        scores.append(score)

    # print(list(sorted(zip(scores, range(len(text))), key=lambda x: -x[0]))[:starting_size])
    # print(list(sorted(zip(scores, range(len(text))), key=lambda x: -x[0]))[:1000])

    # return [x[1] for x in sorted(zip(scores, range(len(text))), key=lambda x: -x[0])][:starting_size]
    return [x[1] for x in sorted(zip(scores, range(len(text))), key=lambda x: -x[0])][:starting_size]

    # sampling_list = []

    # for i, (sent, pos_tag) in enumerate(zip(text, pos_tags)):
    #     total_nnps = sum([1 if 'NNP' in x else 0 for x in pos_tag])
    #     if total_nnps > 2.5 * np.log(len(sent)):
    #         sampling_list.append(i)
    
    # selected_indices = random.sample(sampling_list, starting_size)
    # return selected_indices

