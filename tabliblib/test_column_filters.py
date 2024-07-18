"""
To run tests:
python -m unittest tabliblib/test_column_filters.py -v
"""
import copy
import unittest

import numpy as np
import pandas as pd

from tabliblib.filter.column_filters import InvalidColumnsFilter, MaxColumnsFilter, ColumnFilterChain


class TestColumnFilters(unittest.TestCase):
    def setUp(self) -> None:
        self.df = pd.DataFrame({
            "x": [199, 299, 399],
            "1.456": [199, 299, 399],
            "-3.14": [199, 299, 399],
            "0000": [199, 299, 399],
            "category": ["A", "B", "C"],

        })

    def test_invalid_columns_filter_header_len(self):
        """Test that invalid column filter max_header_len_chars works."""

        column_filter = InvalidColumnsFilter(max_header_len_chars=7,
                                             min_unique_column_values=2,
                                             max_null_like_frac=1.,
                                             )
        out = column_filter(self.df)
        self.assertEqual(out.shape[1], self.df.shape[1] - 1)
        self.assertTrue("category" not in out.columns)

    def test_max_columns_filter(self):
        column_filter = MaxColumnsFilter(max_cols=3)
        out = column_filter(self.df)
        self.assertEqual(out.shape[1], column_filter.max_cols)

    def test_max_columns_filter_no_op(self):
        column_filter = MaxColumnsFilter(max_cols=10)
        out = column_filter(self.df)
        self.assertEqual(out.shape[1], self.df.shape[1])


class TestColumnFilterChain(unittest.TestCase):
    def setUp(self) -> None:
        self.df = pd.DataFrame({
            "x": [199, 299, 399],
            "1.456": [199, 299, 399],
            "-3.14": [199, 299, 399],
            "0000": [199, 199, 399],
            "category": ["A", "B", "C"],

        })

    def test_filter_chain(self):
        """Test that a valid filter chain does not modify the input DataFrame."""
        filter_chain = ColumnFilterChain([
            InvalidColumnsFilter(max_header_len_chars=128,
                                 min_unique_column_values=2,
                                 max_null_like_frac=1.,
                                 ),
            MaxColumnsFilter(max_cols=10)
        ])
        out = filter_chain(self.df)
        pd.testing.assert_frame_equal(self.df, out)

    def test_filter_chain_drops_long_header(self):
        filter_chain = ColumnFilterChain([
            InvalidColumnsFilter(max_header_len_chars=7,
                                 min_unique_column_values=2,
                                 max_null_like_frac=1.,
                                 ),
            MaxColumnsFilter(max_cols=10)
        ])
        out = filter_chain(self.df)
        pd.testing.assert_frame_equal(self.df.drop(columns=["category"]), out)

    def test_filter_chain_drops_insufficient_unique_values(self):
        filter_chain = ColumnFilterChain([
            InvalidColumnsFilter(max_header_len_chars=128,
                                 min_unique_column_values=3,
                                 max_null_like_frac=1.,
                                 ),
            MaxColumnsFilter(max_cols=10)
        ])
        out = filter_chain(self.df)
        pd.testing.assert_frame_equal(self.df.drop(columns=["0000"]), out)

    def test_filter_chain_drops_missing_column(self):
        _df = copy.deepcopy(self.df)
        _df[_df.columns[-1]] = pd.Series([np.nan] * len(_df))
        filter_chain = ColumnFilterChain([
            InvalidColumnsFilter(max_header_len_chars=128,
                                 min_unique_column_values=2,
                                 max_null_like_frac=0.5,
                                 ),
            MaxColumnsFilter(max_cols=10)
        ])
        out = filter_chain(_df)
        pd.testing.assert_frame_equal(_df.drop(columns=[_df.columns[-1]]), out)

    def test_filter_chain_drops_extra_columns(self):
        """Test that a filter chain drops extra columns."""
        max_cols = 4
        filter_chain = ColumnFilterChain([
            InvalidColumnsFilter(max_header_len_chars=128,
                                 min_unique_column_values=2,
                                 max_null_like_frac=1.,
                                 ),
            MaxColumnsFilter(max_cols=max_cols)
        ])
        out = filter_chain(self.df)
        self.assertEqual(out.shape[1], max_cols)
