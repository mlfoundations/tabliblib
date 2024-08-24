from abc import abstractmethod
from dataclasses import dataclass, field
from typing import List, Sequence, Union, Dict, Any, Optional, Callable

import pandas as pd

from tabliblib.filter import Filter, FilterChain
from tabliblib.filter.filter_utils import has_bad_column_headers, is_string_column, compute_frac_contains_code, \
    compute_frac_contains_pii, fetch_names_of_valid_columns
from tabliblib.io import read_arrow_bytes


class TableFilter(Filter):
    """TableFilters either include or exclude a table. They never modify the table."""

    @abstractmethod
    def __call__(self, df: pd.DataFrame) -> bool:
        raise


@dataclass
class TableFilterChain(FilterChain):
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


class ClassifierBasedFilter(TableFilter):
    """Abstract class for classifier-based filtering.

    This should be subclassed."""

    def __call__(self, df: pd.DataFrame) -> bool:
        raise


@dataclass
class TableQualityFilter(ClassifierBasedFilter):
    """Score-based table quality filter."""
    feature_extraction_fn: Callable[[pd.DataFrame], Any]
    classifier: Any
    threshold: float

    def __call__(self, df: pd.DataFrame) -> bool:
        features = self.feature_extraction_fn(df)
        # Reorder the features according to the column ordering at train time
        #  to avoid XGBoost ValueError
        features = features[self.classifier.get_booster().feature_names]
        score = self.classifier.predict_proba(features)
        assert len(score) == 1, f"expected otuput of length 1; got length {len(score)}"
        if len(score.flatten()) == 2:
            score = score.flatten()[-1]
        elif len(score) > 2:
            raise ValueError
        # if score is less than threshold, we drop it.
        # This corresponds to class where a high score indicates "good"
        # and a low score indicates "bad" (i.e. data we want to remove).
        return score > self.threshold


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
