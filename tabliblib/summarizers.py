import re
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, Any, List, Optional, Set, Callable, Sequence

import numpy as np
import pandas as pd
from nltk import word_tokenize
from scipy.stats import entropy
from scipy.stats import pearsonr
from sklearn.feature_selection import mutual_info_regression


@dataclass
class ColumnSummarizer(ABC):
    prefix: str
    output_keys: Optional[Set[str]] = None

    def __post_init__(self):
        self._init_output_keys()

    def _set_output_keys(self, keys: Set[str]) -> None:
        assert self.output_keys is None
        self.output_keys = keys

    def generic_summarize(self, col: pd.Series) -> Dict[str, float]:
        """Generic summary stats used by all ColumnSummarizers for any dtype."""
        missing_count = col.isnull().sum()
        missing_frac = missing_count / len(col)

        # Calculate value counts
        value_counts = col.value_counts()
        total_count = len(col) - missing_count

        # Most frequent stats
        most_frequent_count = value_counts.iloc[0] if not value_counts.empty else 0
        most_frequent_frac = most_frequent_count / total_count if total_count > 0 else 0

        # Least frequent stats
        least_frequent_count = value_counts.iloc[-1] if len(value_counts) > 0 else 0
        least_frequent_frac = least_frequent_count / total_count if total_count > 0 else 0

        # Calculate entropy
        probabilities = value_counts / total_count
        col_entropy = entropy(probabilities) if total_count > 0 else 0

        return {
            "missing": missing_count,
            "missing_frac": missing_frac,
            "most_frequent_count": most_frequent_count,
            "most_frequent_frac": most_frequent_frac,
            "least_frequent_count": least_frequent_count,
            "least_frequent_frac": least_frequent_frac,
            "entropy": col_entropy
        }

    @abstractmethod
    def _init_output_keys(self):
        """Call on some dummy data to init the output keys."""
        raise

    @abstractmethod
    def __call__(self, col: pd.Series) -> Dict[str, float]:
        raise


@dataclass
class NumericColumnSummarizer(ColumnSummarizer):
    prefix: str = "num_"

    def _init_output_keys(self):
        self.__call__(pd.Series(np.arange(10)))

    def _check_inputs(self, col: pd.Series) -> None:
        if not pd.api.types.is_numeric_dtype(col):
            raise TypeError("Input series must be of numeric type")

    def __call__(self, col: pd.Series) -> Dict[str, float]:
        self._check_inputs(col)

        # Remove NaN values
        col_clean = col.dropna()

        # Calculate statistics
        mean = np.mean(col_clean)
        std = np.std(col_clean, ddof=1)

        outputs_raw = {
            'kurtosis': pd.Series.kurtosis(col_clean),
            'skewness': pd.Series.skew(col_clean),
            'mad': np.mean(np.abs(col_clean - mean)),
            'std': std,
            **self.generic_summarize(col)
        }
        output = {self.prefix + k: v for k, v in outputs_raw.items()}
        if not self.output_keys:
            self._set_output_keys(set(output.keys()))
        return output


@dataclass
class BoolColumnSummarizer(ColumnSummarizer):
    prefix: str = "bool_"

    def _init_output_keys(self):
        self.__call__(pd.Series(np.random.uniform(size=10) > 1))

    def _check_inputs(self, col: pd.Series) -> None:
        if not pd.api.types.is_bool_dtype(col):
            raise TypeError("Input series must be of boolean type")

    def __call__(self, col: pd.Series) -> Dict[str, float]:
        self._check_inputs(col)

        # Calculate the mean of non-null values
        bool_mean = col.mean()

        outputs_raw = {
            'bool_mean': bool_mean,
            **self.generic_summarize(col)
        }
        output = {self.prefix + k: v for k, v in outputs_raw.items()}
        if not self.output_keys:
            self._set_output_keys(set(output.keys()))
        return output


@dataclass
class ObjectColumnSummarizer(ColumnSummarizer):
    prefix: str = "obj_"

    def _init_output_keys(self):
        self.__call__(pd.Series(np.random.choice(["a", "b", "c"], 10)))

    def _check_inputs(self, col: pd.Series) -> None:
        if not pd.api.types.is_object_dtype(col):
            raise TypeError("Input series must be of object type")

    def __call__(self, col: pd.Series) -> Dict[str, Any]:
        self._check_inputs(col)

        # Convert all elements to strings and get their lengths
        lengths = col.astype(str).str.len()

        outputs_raw = {
            'n_unique': col.nunique(),
            'min_length': lengths.min(),
            'max_length': lengths.max(),
            'mean_length': lengths.mean(),
            'median_length': lengths.median(),
            **self.generic_summarize(col)
        }
        output = {self.prefix + k: v for k, v in outputs_raw.items()}
        if not self.output_keys:
            self._set_output_keys(set(output.keys()))
        return output


from datetime import datetime, timedelta


def generate_random_dates(start_date=datetime(2000, 1, 1),
                          end_date=datetime(2023, 12, 31)):
    date_range = (end_date - start_date).days
    random_days = np.random.randint(0, date_range, 10)
    random_dates = [start_date + timedelta(days=int(day)) for day in random_days]
    return random_dates


@dataclass
class DateTimeColumnSummarizer(ColumnSummarizer):
    prefix: str = "dt_"

    def _init_output_keys(self):
        self.__call__(pd.Series(generate_random_dates()))

    def _check_inputs(self, col: pd.Series) -> None:
        if not pd.api.types.is_datetime64_any_dtype(col):
            raise TypeError("Input series must be of datetime type")

    def __call__(self, col: pd.Series) -> Dict[str, float]:
        self._check_inputs(col)

        # Sort the series and drop NaT values
        col_clean = col.sort_values().dropna()

        if len(col_clean) > 1:
            time_span = col_clean.max() - col_clean.min()
            time_span_secs = time_span.total_seconds()
            time_span_days = time_span.total_seconds() / (24 * 3600)
            time_span_years = time_span_days / 365.25

            # Calculate average time difference between consecutive observations
            timedeltas = col_clean.diff()[1:]  # Skip the first NaT difference
            timedelta_mean = timedeltas.mean().total_seconds()
        else:
            time_span_secs = time_span_days = time_span_years = timedelta_mean = 0

        outputs_raw = {
            'time_span_secs': time_span_secs,
            'time_span_days': time_span_days,
            'time_span_years': time_span_years,
            'timedelta_mean': timedelta_mean,
            'is_monotonic': float(col.is_monotonic_increasing or col.is_monotonic_decreasing),
            **self.generic_summarize(col)
        }
        output = {self.prefix + k: v for k, v in outputs_raw.items()}
        if not self.output_keys:
            self._set_output_keys(set(output.keys()))
        return output


class HeaderSummarizer:
    """Summarizes the (text) headers of a table."""

    def __init__(self):
        self.output_keys: Set[str] = None
        self._init_output_keys("a_column")

    def _init_output_keys(self, colname):
        result = self.__call__(colname)
        self.output_keys = set(result.keys())

    def __call__(self, colname: str) -> Dict[str, float]:
        if not isinstance(colname, str):
            print(f"[warning] casting column name {colname} to str from {type(colname)}")
            colname = str(colname)
        return {"header_num_chars": len(colname),
                "header_num_underscore": len(re.findall("_", colname)),
                "header_num_words": len(word_tokenize(colname)),
                }


class MultiColumnSummarizer:
    """Compute pairwise metrics between columns."""

    def __init__(self, max_n=512):
        self.max_n = max_n

        dummy_data = pd.DataFrame({
            "x": np.random.uniform(size=10),
            "y": np.random.uniform(size=10),
        })
        _tmp_output = self.__call__(dummy_data)
        self.output_keys = set(_tmp_output.keys())

    def __call__(self, df: pd.DataFrame) -> Dict[str, List[float]]:
        # Downsample if necessary
        if len(df) > self.max_n:
            df = df.sample(n=self.max_n, random_state=42)

        # Select only numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        pearson_correlations = []
        mutual_informations = []

        # Iterate over pairs of numeric columns
        for col1, col2 in combinations(numeric_cols, 2):
            x = df[col1].values
            y = df[col2].values

            # Remove rows where either column has NaN
            mask = ~(np.isnan(x) | np.isnan(y))
            x = x[mask]
            y = y[mask]

            if len(x) > 1:  # Need at least 2 samples for correlation
                # Compute Pearson correlation
                corr, _ = pearsonr(x, y)
                pearson_correlations.append(corr)

                # Compute mutual information
                mi = mutual_info_regression(x.reshape(-1, 1), y)[0]
                mutual_informations.append(mi)

                # TODO(jpgard): add covariances

        return {
            "pearson_correlation": pearson_correlations,
            "mutual_information": mutual_informations
        }


@dataclass
class TableSummarizer:
    numeric_summarizer: NumericColumnSummarizer = NumericColumnSummarizer()
    bool_summarizer: BoolColumnSummarizer = BoolColumnSummarizer()
    obj_summarizer: ObjectColumnSummarizer = ObjectColumnSummarizer()
    datetime_summarizer: DateTimeColumnSummarizer = DateTimeColumnSummarizer()
    # multicolumn_summarizer: MultiColumnSummarizer = MultiColumnSummarizer()
    header_summarizer: HeaderSummarizer = HeaderSummarizer()
    agg_quantiles:Optional[Sequence[float]] = (0.1, 0.25, 0.75, 0.9)
    agg_fns: Dict[str, Callable[[Sequence[Any]], float]] = field(default_factory=lambda: {
        "min": np.nanmin,
        "max": np.nanmax,
        "mean": np.nanmean,
        "median": np.nanmedian,
    })
    include_table_summary_metrics: bool = True

    @property
    def summarizers_to_check_keys(self):
        """A list of the summarizers whose keys should always be included in output."""
        return [self.numeric_summarizer,
                self.bool_summarizer,
                self.obj_summarizer,
                self.datetime_summarizer,
                # self.multicolumn_summarizer,
                self.header_summarizer]

    def summarize_table(self, df) -> Dict[str, float]:
        """Global summaries that do not need to be aggregated."""
        prefix = "table_"
        duplicate_count = df.duplicated().sum()
        dtype_count = df.dtypes.nunique()
        output = {
            "duplicate_count": duplicate_count,
            "duplicate_frac": duplicate_count / len(df),
            "dtype_unique_count": dtype_count,
            "n": len(df),
            "ncols": df.shape[1],
        }

        def _column_fn(df, fn):
            return sum(fn(df[c]) for c in df.columns)

        output["dtype_numeric_count"] = _column_fn(df, pd.api.types.is_numeric_dtype)
        output["dtype_bool_count"] = _column_fn(df, pd.api.types.is_bool_dtype)
        output["dtype_object_count"] = _column_fn(df, pd.api.types.is_object_dtype)
        output["dtype_datetime_count"] = _column_fn(df, pd.api.types.is_datetime64_any_dtype)
        for dtype in ("numeric", "bool", "object", "datetime"):
            output[f"dtype_{dtype}_frac"] = output[f"dtype_{dtype}_count"] / df.shape[1]

        return {prefix + k: v for k, v in output.items()}

    def _compute_column_summary_stats(self, df: pd.DataFrame) -> Dict[str, List]:
        stats = defaultdict(list)
        for colname in df.columns:
            col = df[colname]
            if pd.api.types.is_numeric_dtype(col):
                col_summary = self.numeric_summarizer(col)
            elif pd.api.types.is_bool_dtype(col):
                col_summary = self.bool_summarizer(col)
            elif pd.api.types.is_object_dtype(col):
                col_summary = self.obj_summarizer(col)
            elif pd.api.types.is_datetime64_any_dtype(col):
                col_summary = self.datetime_summarizer(col)
            else:
                raise ValueError(f"warning: got unhandled dtype {col.dtype}")
            col_summary.update(self.header_summarizer(colname))
            for k, v in col_summary.items():
                stats[k].append(v)
        return stats

    @property
    def expected_keys(self) -> List[str]:
        expected_keys = []
        for summarizer in self.summarizers_to_check_keys:
            for output_col in summarizer.output_keys:
                for quantile in self.agg_quantiles:
                    expected_keys.append(f"{output_col}_quantile{quantile}")
                for agg_func in self.agg_fns.keys():
                    expected_keys.append(f"{output_col}_{agg_func}")
        return expected_keys

    def __call__(self, df: pd.DataFrame) -> pd.Series:
        stats = self._compute_column_summary_stats(df)

        result = {}

        # Compute min, median, mean, max for all collected stats
        for stat, values in stats.items():
            for agg_name, agg_fn in self.agg_fns.items():
                result[f"{stat}_{agg_name}"] = agg_fn(values)

            for quantile in self.agg_quantiles:
                result[f"{stat}_quantile{quantile}"] = np.nanquantile(values, q=quantile)

        # Ensure that every column is in the output.
        expected_keys = self.expected_keys
        for expected_key in expected_keys:
            if expected_key not in result.keys():
                result[expected_key] = np.nan

        features_after_check = len(result)
        assert features_after_check == len(expected_keys)

        if self.include_table_summary_metrics:
            # add table-level features
            result.update(self.summarize_table(df))

        return pd.Series(result)

class SingleColumnSummarizer(TableSummarizer):

    @property
    def expected_keys(self) -> List[str]:
        expected_keys = []
        for summarizer in self.summarizers_to_check_keys:
            for output_col in summarizer.output_keys:
                expected_keys.append(output_col)
        return expected_keys
    def __call__(self, ser:pd.Series) -> pd.Series:
        assert isinstance(ser, pd.Series), f"expected type pd.Series, got {type(ser)}"
        df = pd.DataFrame(ser)
        stats = self._compute_column_summary_stats(df)
        stats = {k:v[0] for k,v in stats.items()}


        # Ensure that every column is in the output.
        expected_keys = self.expected_keys
        for expected_key in expected_keys:
            if expected_key not in stats.keys():
                stats[expected_key] = np.nan

        features_after_check = len(stats)
        assert features_after_check == len(expected_keys)

        return pd.Series(stats)


