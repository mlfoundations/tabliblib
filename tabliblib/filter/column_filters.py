from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Union, List, Sequence, Dict, Any

import pandas as pd

from tabliblib.filter import Filter, FilterChain
from tabliblib.filter.filter_utils import fetch_names_of_valid_columns
from tabliblib.io import read_arrow_bytes, sample_columns_if_needed


@dataclass
class ColumnFilter(Filter):
    """ColumnFilters modify a table based on column-level filtering criteria.

    Valid columns are retained and invalid columns are dropped.
    When the column filtering results in an empty dataframe, None is returned."""

    @abstractmethod
    def __call__(self, df: pd.DataFrame) -> Union[pd.DataFrame, None]:
        raise


@dataclass
class ColumnFilterChain(FilterChain):
    """A chain of ColumnFilters, applied sequentially."""
    _chain: List[ColumnFilter] = field(default_factory=list)

    def append(self, flt: ColumnFilter):
        self._chain.append(flt)

    def extend(self, filters: Sequence[ColumnFilter]) -> None:
        for flt in filters:
            assert isinstance(flt, ColumnFilter), f"expected ColumnFilter, got type {type(flt)}"
            self._chain.append(flt)

    def __call__(self, elem: Union[pd.DataFrame, None, Dict[str, Any]],
                 dict_key="arrow_bytes",
                 parse_arrow_bytes_from_dict: bool = True) -> Union[pd.DataFrame, None]:
        """Apply the filters to elem.

        Elem can be a DataFrame, a dictionary containing a DataFrame or Arrow bytes
        as a value under dict_key, or None (in which case the chain will return False).
        """
        if elem is None:
            return None
        if isinstance(elem, Dict):
            df = elem[dict_key]
            if parse_arrow_bytes_from_dict:
                df = read_arrow_bytes(df)
        else:
            df = elem

        for filter_obj in self._chain:
            df = filter_obj(df)
            if not len(df):
                return None
        return df


@dataclass
class InvalidColumnsFilter(ColumnFilter):
    max_header_len_chars: int
    min_unique_column_values: int
    max_null_like_frac: float

    def __call__(self, df: pd.DataFrame) -> Union[pd.DataFrame, None]:
        valid_cols = fetch_names_of_valid_columns(
            df,
            max_header_len_chars=self.max_header_len_chars,
            min_unique_column_values=self.min_unique_column_values,
            max_null_like_frac=self.max_null_like_frac)
        return df[valid_cols]


@dataclass
class MaxColumnsFilter(ColumnFilter):
    max_cols: int

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        return sample_columns_if_needed(df, self.max_cols)
