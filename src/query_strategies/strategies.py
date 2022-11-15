import random
from typing import List


def random_query(predictions: List[List[float]], k=5):
    random.choices(list(range(len(predictions))), k=k)
