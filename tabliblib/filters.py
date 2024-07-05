import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable, Union, Sequence

import numpy as np
import pandas as pd

from tabliblib.config import PreprocessConfig
from tabliblib.io import read_arrow_bytes


class Filter(ABC):
    """Generic class to represent a filter."""

    def __call__(self, *args, **kwargs):
        raise


class TableFilter(Filter):
    """TableFilters either include or exclude a table. They never modify the table."""

    @abstractmethod
    def __call__(self, df: pd.DataFrame) -> bool:
        raise


@dataclass
class TableFilterChain(ABC):
    """A chain of TableFilters, applied sequentially."""
    _chain: List[TableFilter] = field(default_factory=list)

    def append(self, filter: TableFilter):
        self._chain.append(filter)

    def extend(self, filters: Sequence[TableFilter]) -> None:
        for flt in filters:
            assert isinstance(flt, TableFilter), f"expected TableFilter, got type {type(flt)}"
            self._chain.append(flt)

    def __call__(self, elem: Union[pd.DataFrame, None, Dict[str, Any]],
                 dict_key="arrow_bytes",
                 parse_arrow_bytes_from_dict: bool = True) -> bool:
        """Apply the filters to elem.
        
        If False, this means the table should be excluded (the filters 'triggered').
        If True, this means the table should be retrained (the filters did not 'trigger').

        Elem can be a DataFrame, a dictionary containing a DataFrame or Arrow bytes
        as a value under dict_key, or None (in which case the chain will return False).
        """
        if elem is None:
            return False
        if isinstance(elem, Dict):
            df = elem[dict_key]
            if parse_arrow_bytes_from_dict:
                df = read_arrow_bytes(df)
        else:
            df = elem

        for filter_obj in self._chain:
            if not filter_obj(df):
                return False
        return True


def convert_bytes_to_string(byte_sequence) -> str:
    # List of encodings to try
    encodings = ['utf-8', 'ascii', 'iso-8859-1', 'utf-16', 'utf-32']

    # Iterate through each encoding
    for encoding in encodings:
        try:
            # Attempt to decode the byte sequence
            decoded_string = byte_sequence.decode(encoding)
            # If successful, return the string and encoding used
            return decoded_string
        except UnicodeDecodeError:
            # If decoding fails, continue to the next encoding
            continue

    raise ValueError(f"could not encode object of type {type(byte_sequence)}: {byte_sequence}")


def cast_to_str(s) -> str:
    if isinstance(s, str):
        return s
    if isinstance(s, bytes):
        return convert_bytes_to_string(s)
    else:
        return str(s)


def is_numeric(s) -> bool:
    """Check whether a string is numeric. This includes floats such as '3.5' and 3.'."""
    if not isinstance(s, str):
        s = cast_to_str(s)
    return bool(re.match(r"^-?\d+(\.+\d+)?$", s))


def contains_code(text) -> bool:
    """Test whether a string contains code.

    This functions uses a set of heuristics to determine whether a string contains code in any common languages.
    It is likely that it fails to detect many cases; it is best to use this is a noisy signal over many cells
    (i.e. threshold whether the fraction of cells in a colum where code is detected is > 0.5) rather than using
    the function applied to a single instance as a decision rule.
    """
    if not isinstance(text, str):
        text = cast_to_str(text)
    # Regular expressions for different code-like patterns
    patterns = [
        # r'\b(def|class|import|from|export|const|let|var|if|else|while|for|switch|case|break|continue|return|try|catch|finally|function|namespace|using)\b',  # Common keywords
        r'\b(def\s+\w+\s*\([^)]*\)\s*:|function\s+\w+\s*\(|\w+\s*\([^)]*\)\s*{)',  # function definitions
        r'\/\*[\s\S]*?\*\/|#.*',  # Single-line or multi-line comments in various languages
        r'\b(System\.out\.println|console\.log|print|echo|printf|scanf|cin|cout)\b',
        # Print statements in various languages
        r'(?<=\s)(int|char|float|double|string|bool|boolean|void|public|private|protected|static|final|class)\s+(?=\w+\s*[;=(){}])'
        r'[^a-zA-Z](=|\+=|\-=|\*=|\/=|==|!=|<=|>=|<|>|\|\||&&|!|\^|\||&|\+|-|\*|\/|\%|\+\+|\-\-);?',
        # Assignment and comparison operators, and arithmetic operations
        # r'\b(try|catch|finally|throw|throws|new|delete|self|super)\b',  # Exception handling and other object-oriented keywords
        # r'\{|\}|\(|\)|\[|\];?',  # Common punctuation (braces, parentheses, square brackets)
        r'->|::|\.\.|=>',  # Language-specific operators (C++, PHP, Python, JavaScript)
    ]

    # Check if the text matches any of the code-like patterns
    for pattern in patterns:
        if re.search(pattern, text):
            return True

    return False


def contains_pii(text) -> bool:
    """Test whether a string contains PII (email number of phone numbers).

    Note that this can only
    """
    if not isinstance(text, str):
        text = cast_to_str(text)

    # Regular expression pattern that matches most email addresses and phone numbers,
    # courtesy of ChatGPT.
    # Regular expression pattern that matches most email addresses and specifically formatted phone numbers
    pattern = re.compile(
        r'(\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b)|'  # Email pattern
        r'((?<!\d)(?:\(\d{1,4}\)\s?|\d{1,4}[-.\s])\d{1,4}[-.\s]\d{1,4}(?!\d))'
        # Phone pattern with formatting (ignores simple digit strings)
        , re.IGNORECASE)
    # Search text for matches
    match = pattern.search(text)
    # Return True if a match is found, False otherwise
    return bool(match)


def is_english(row, threshold=0.5) -> bool:
    langdetect_result = row.get("langdetect_result")
    if not langdetect_result:
        return False

    return langdetect_result.get("lang") == "en" and langdetect_result.get("score") >= threshold


def is_kv_header(x) -> bool:
    x = cast_to_str(x)
    return re.search('\".*\".*:.*\".*\"', x) is not None or re.search("\'.*\'.*:.*\'.*\'", x) is not None


def compute_frac_numeric_colnames(df) -> float:
    return np.mean([is_numeric(x) for x in df.columns.values])


def has_bad_column_headers(df: pd.DataFrame,
                           max_frac_numeric_colnames: Union[float, None],
                           max_frac_unnamed_columns: Union[float, None]) -> bool:
    # Note: headers can be of numeric types (i.e. np.float64, etc) and bytes type so
    # they must be explicitly cast to string first.
    if str(df.columns[0]).startswith("{") and str(df.columns[-1]).endswith("}"):
        return True

    if (max_frac_numeric_colnames is not None
            and compute_frac_numeric_colnames(df) > max_frac_numeric_colnames):
        return True

    if max_frac_unnamed_columns is not None and np.mean(
            ["Unnamed:" in cast_to_str(c) for c in df.columns]) > max_frac_unnamed_columns:
        return True

    if any(is_kv_header(x) for x in df.columns):
        # Case: this is probably a key-value pair, e.g. ' "landscape_name": "GB1_combo"'
        return True

    return False


def safe_nunique(x: pd.Series) -> int:
    """Safe wrapper for pd.Series.nunique() that handles unhashable types."""
    try:
        return x.nunique()
    except TypeError as te:
        if "unhashable" in str(te):
            return x.astype(str).nunique()
        else:
            raise te


def is_string_column(x: pd.Series) -> bool:
    return x.dtype == 'object' or pd.api.types.is_string_dtype(x)


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


def compute_frac_null_like(x: pd.Series) -> float:
    """Compute the fraction of observations that are nan, empty, None.

    Used to threshold columns, for example filter columns with < x null-like values.
    """
    is_null = pd.isnull(x)

    def is_empty_string_or_whitespace(x) -> bool:
        return x == "" or (re.search("^\s+$", str(x)) is not None)

    is_empty_string = x.apply(is_empty_string_or_whitespace)
    return np.mean(np.logical_or(is_null.values, is_empty_string.values))


def compute_frac_contains_code(x: pd.Series) -> float:
    return x.apply(contains_code).astype(float).mean()


def compute_frac_contains_pii(x: pd.Series) -> float:
    return x.apply(contains_pii).astype(float).mean()


def fetch_names_of_valid_columns(df, max_header_len_chars: int,
                                 min_unique_column_values: int,
                                 max_null_like_frac: Optional[float] = None) -> List[str]:
    """Determine whether a column is valid.

    Only valid columns (those passing this test) should be included in the final dataset."""
    return [x for x in df.columns.tolist()
            if x is not None
            and len(str(x)) < max_header_len_chars
            and safe_nunique(df[x]) >= min_unique_column_values
            and (compute_frac_null_like(df[x]) < max_null_like_frac if max_null_like_frac is not None else True)]


@dataclass
class RowCountFilter(TableFilter):
    min_rows: int
    max_rows: Optional[int] = None

    def __call__(self, df: pd.DataFrame) -> bool:
        if len(df) < self.min_rows:
            return False
        elif self.max_rows is not None and len(df) > self.max_rows:
            return False
        else:
            return True


@dataclass
class ColumnCountFilter(TableFilter):
    min_columns: int
    max_columns: Optional[int] = None

    def __call__(self, df: pd.DataFrame) -> bool:
        if len(df.columns) < self.min_columns:
            return False
        elif self.max_columns is not None and len(df.columns) > self.max_columns:
            return False
        return True

@dataclass
class BadHeadersFilter(TableFilter):
    max_frac_numeric_colnames: Optional[float] = None
    max_frac_unnamed_columns: Optional[float] = None

    def __call__(self, df: pd.DataFrame) -> bool:
        if has_bad_column_headers(df, max_frac_numeric_colnames=self.max_frac_numeric_colnames,
                                  max_frac_unnamed_columns=self.max_frac_unnamed_columns):
            return False
        return True


@dataclass
class SchemaFilter(TableFilter):
    """Filter based on the schema of the table."""
    min_dtypes: Optional[int] = None

    def __call__(self, df: pd.DataFrame) -> bool:
        if self.min_dtypes is not None and len(df.dtypes.value_counts()) < self.min_dtypes:
            return False
        return True


@dataclass
class ValidColumnCountFilter(TableFilter):
    max_header_len_chars: int
    min_unique_column_values: int
    max_null_like_frac: float
    min_cols: int

    def __call__(self, df: pd.DataFrame) -> bool:
        valid_cols = fetch_names_of_valid_columns(
            df,
            max_header_len_chars=self.max_header_len_chars,
            min_unique_column_values=self.min_unique_column_values,
            max_null_like_frac=self.max_null_like_frac)

        if len(valid_cols) < self.min_cols:
            return False
        return True


@dataclass
class CodeDetectionFilter(TableFilter):
    """Designed to drop tables containing code.
    This is a frequent occurrence in TabLib, and therefore
    necessitates its own table-level filter to aggressively remove code."""
    code_detect_filter_threshold: Optional[float] = None

    def __call__(self, df: pd.DataFrame) -> bool:
        string_colnames = [c for c in df.columns if is_string_column(df[c])]
        if self.code_detect_filter_threshold is not None and any(
                compute_frac_contains_code(df[c]) > self.code_detect_filter_threshold for c in string_colnames):
            return False
        return True


@dataclass
class PIIDetectionFilter(TableFilter):
    """Designed to drop tables containing PII."""
    pii_detect_filter_threshold: Optional[float] = None

    def __call__(self, df: pd.DataFrame) -> bool:
        string_colnames = [c for c in df.columns if is_string_column(df[c])]
        if self.pii_detect_filter_threshold is not None and any(
                compute_frac_contains_pii(df[c]) for c in string_colnames
        ):
            return False
        return True
