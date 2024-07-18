import logging
from abc import abstractmethod
from dataclasses import dataclass
from typing import Callable, Any, Union, Sequence

import pandas as pd

from tabliblib.filter import Filter, FilterChain
from tabliblib.filter.filter_utils import is_string_column


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
    pass


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
