import logging
from typing import Callable, Any

import pandas as pd

from tabliblib.filter.filter_utils import is_string_column


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
