import logging
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Any, Union, Sequence, List, Dict

import pandas as pd

from tabliblib.filter import Filter, FilterChain
from tabliblib.filter.filter_utils import is_string_column, contains_code, contains_pii
from tabliblib.io import read_arrow_bytes


@dataclass
class RowFilter(Filter):
    """RowFilters modify a table based on row-level filtering criteria.

    Valid rows are retained and invalid rows are dropped.
    When the row filtering results in an empty DataFrame, None is returned.
    """

    @abstractmethod
    def __call__(self, df: pd.DataFrame) -> Union[pd.DataFrame, None]:
        raise

    def _apply_row_based_filter(self,
                                df: pd.DataFrame,
                                filter_fn: Callable[[Any], bool],
                                string_columns_only=False
                                ) -> pd.DataFrame:
        """Apply filter_fn to string columns in the dataset.

        :param df: Dataframe to apply to.
        :param filter_fn: Function to apply to every cell where the column is of a string dtype.
            filter_fn should return True if the row should be dropped. For details on what columns
            are considered string types, see is_string_column.

        :return: Dataframe with the filtered rows removed.
        """
        # Initialize a mask for rows to keep (all True initially)
        keep_rows_mask = pd.Series(True, index=df.index)

        # Iterate through each column
        for column in df.columns:
            # Check if the column is of type object or string
            if (not string_columns_only) or is_string_column(df[column]):
                should_be_dropped = df[column].apply(filter_fn)

                # Update the keep_rows_mask: if should_be_dropped is True,
                # set the corresponding row in keep_rows_mask to False
                keep_rows_mask &= ~should_be_dropped

        if keep_rows_mask.sum() < len(df):
            logging.warning(
                f"dropping {(~keep_rows_mask).sum()} rows in apply_row_based_filter ({keep_rows_mask.sum() / len(df):.4f} fraction of input rows)")

        # Filter the DataFrame based on the keep_rows_mask and return the result
        return df[keep_rows_mask]


@dataclass
class RowFilterChain(FilterChain):
    """A chain of RowFilters, applied sequentially."""
    _chain: List[RowFilter] = field(default_factory=list)

    def append(self, flt: RowFilter):
        self._chain.append(flt)

    def extend(self, filters: Sequence[RowFilter]) -> None:
        for flt in filters:
            assert isinstance(flt, RowFilter), f"expected RowFilter, got type {type(flt)}"
            self._chain.append(flt)

    def __call__(self, elem: Union[pd.DataFrame,
    None, Dict[str, Any]],
                 dict_key="arrow_bytes",
                 parse_arrow_bytes_from_dict: bool = True
                 ) -> Union[pd.DataFrame, None]:
        """Apply the filters to elem.

        If False, this means the table should be excluded (the filters 'triggered').
        If True, this means the table should be retrained (the filters did not 'trigger').

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
class MaxValueLengthFilter(RowFilter):
    """Filter out any rows where any cell value exceeds the max length, in characters.

    Numeric columns are ignored.
    """
    max_value_len_chars: int
    string_columns_only: bool = True

    def __call__(self, df: pd.DataFrame) -> Union[pd.DataFrame, None]:
        return self._apply_row_based_filter(
            df,
            filter_fn=lambda x: len(str(x)) > self.max_value_len_chars,
            string_columns_only=self.string_columns_only)


@dataclass
class MaxRowCountFilter(RowFilter):
    max_rows: int
    """If there are too many rows, randomly downsample them (without replacement)."""

    def __call__(self, df: pd.DataFrame) -> Union[pd.DataFrame, None]:
        if len(df) <= self.max_rows:
            return df
        else:
            return df.sample(n=self.max_rows, replace=False)


@dataclass
class SubstringFilter(RowFilter):
    """Filter out any rows where cell contains any of the substrings."""
    substrings: Sequence[str]
    string_columns_only: bool = False

    def __post_init__(self):
        if not (isinstance(self.substrings, tuple)
                or isinstance(self.substrings, list)) \
                or isinstance(self.substrings, str):
            raise ValueError("substrings must be a tuple or list, NOT a string.")

    def __call__(self, df: pd.DataFrame) -> Union[pd.DataFrame, None]:
        def _contains_substring_filter_fn(x) -> bool:
            """Helper function to check if x contains substring. Returns False if x is not castable to string."""
            try:
                return any(substr in str(x) for substr in self.substrings)
            except:
                return False

        return self._apply_row_based_filter(df,
                                            _contains_substring_filter_fn,
                                            string_columns_only=self.string_columns_only)


@dataclass
class CodeRegexFilter(RowFilter):
    """Drop rows with cell values that match a regular expression designed to detect code.

    Only string columns are evaluated."""

    def __call__(self, df: pd.DataFrame) -> Union[pd.DataFrame, None]:
        return self._apply_row_based_filter(df, filter_fn=contains_code,
                                            string_columns_only=True)


@dataclass
class PIIRegexFilter(RowFilter):
    """Drop rows with cell values that match a regular expression designed to detect PII.

        Only string columns are evaluated."""

    def __call__(self, df: pd.DataFrame) -> Union[pd.DataFrame, None]:
        return self._apply_row_based_filter(df, filter_fn=contains_pii,
                                            string_columns_only=True)


@dataclass
class DuplicateRowsFilter(RowFilter):
    def __call__(self, df: pd.DataFrame) -> Union[pd.DataFrame, None]:
        return df.drop_duplicates()


def apply_row_based_filter(df: pd.DataFrame, filter_fn: Callable[[Any], bool],
                           string_columns_only=False) -> pd.DataFrame:
    """Apply filter_fn to string columns in the dataset.

    :param df: Dataframe to apply to.
    :param filter_fn: Function to apply to every cell where the column is of a string dtype.
        filter_fn should return True if the row should be dropped. For details on what columns
        are considered string types, see is_string_column.

    :return: Dataframe with the filtered rows removed.
    """
    # Initialize a mask for rows to keep (all True initially)
    keep_rows_mask = pd.Series(True, index=df.index)

    # Iterate through each column
    for column in df.columns:
        # Check if the column is of type object or string
        if (not string_columns_only) or is_string_column(df[column]):
            should_be_dropped = df[column].apply(filter_fn)

            # Update the keep_rows_mask: if should_be_dropped is True,
            # set the corresponding row in keep_rows_mask to False
            keep_rows_mask &= ~should_be_dropped

    if keep_rows_mask.sum() < len(df):
        logging.warning(
            f"dropping {(~keep_rows_mask).sum()} rows in apply_row_based_filter ({keep_rows_mask.sum() / len(df):.4f} fraction of input rows)")

    # Filter the DataFrame based on the keep_rows_mask and return the result
    return df[keep_rows_mask]
