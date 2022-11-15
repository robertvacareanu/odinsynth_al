"""
This file contains the associated code for querying
That is, given a list of datapoints, each with an 
associated predictions (here modelled as a list
of floats), we return a list of indices, corresponding
to the indices we want to select for annotations (as per AL
paradigm))
"""
import random
from typing import List


"""
In this query implementation we just select random
"""
def random_query(predictions: List[List[float]], k=5) -> List[int]:
    random.choices(list(range(len(predictions))), k=k)
