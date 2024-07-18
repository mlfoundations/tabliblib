from abc import ABC, abstractmethod
from typing import Sequence, Union, Dict, Any

import pandas as pd


class Filter(ABC):
    """Generic class to represent a filter."""

    def __call__(self, *args, **kwargs):
        raise


class FilterChain(ABC):

    @abstractmethod
    def append(self, filter: Filter):
        raise

    @abstractmethod
    def extend(self, filters: Sequence[Filter]) -> None:
        raise

    @abstractmethod
    def __call__(self, elem: Union[pd.DataFrame, None, Dict[str, Any]],
                 dict_key="arrow_bytes",
                 parse_arrow_bytes_from_dict: bool = True) -> Union[pd.DataFrame, None]:
        raise
