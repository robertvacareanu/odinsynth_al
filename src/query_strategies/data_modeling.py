"""
Data modeling for query strategies
Concretely, we might want more information from a query strategy
This class allows us to wrap everything we want/need here

Regarding the design choice of using a `@dataclass` instead of a Dict,
a dataclass allows us to include docstrings for the variables, together
with type annotations
"""

from dataclasses import dataclass
from typing import Union, Tuple, List

from src.utils import ALAnnotation

@dataclass
class QueryStrategyOutput:
    selections             : Union[List[int], List[Tuple[int, List[int]]]]
    """What was selected to be annotated"""
    
    selected_dataset_so_far: List[Tuple[int, ALAnnotation]]
    """The dataset selected so far"""

    resulting_dataset      : List[Tuple[int, ALAnnotation]]
    """The dataset after we added the current selections"""
