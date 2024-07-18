"""
Tests for row filters.

To run tests:
python -m unittest tabliblib/test_row_filters.py -v
"""
import unittest

import pandas as pd

from tabliblib.filter.row_filters import SubstringFilter, MaxValueLengthFilter


class TestMaxValueLengthFilter(unittest.TestCase):
    def setUp(self) -> None:
        self.df = pd.DataFrame({
            "x": [199, 299, 399],
            "1.456": [199, 299, 399],
            "-3.14": [199, 299, 399],
            "0000": [199, 199, 399],
            "category": ["A" * 10, "B", "C"],

        })

    def test_max_value_length_filter(self):
        row_filter = MaxValueLengthFilter(max_value_len_chars=9)
        out = row_filter(self.df)
        self.assertEqual(len(out), len(self.df) - 1)

    def test_max_value_length_no_filter(self):
        """Test that MaxValueLengthFilter does not remove any rows when none are too long."""
        row_filter = MaxValueLengthFilter(max_value_len_chars=1024)
        out = row_filter(self.df)
        self.assertEqual(len(out), len(self.df))


class TestSubstringFilter(unittest.TestCase):
    def setUp(self) -> None:
        self.df = pd.DataFrame({
            "x": [199, 299, 399],
            "1.456": [199, 299, 399],
            "-3.14": [199, 299, 399],
            "0000": [199, 199, 399],
            "category": ["exclude_me", "keep_me", "keep_me"],

        })

    def test_substring_filter(self):
        row_filter = SubstringFilter(substrings=["exclude_me"])
        out = row_filter(self.df)
        pd.testing.assert_frame_equal(out, self.df.iloc[1:])

    def test_substring_filter_raises_on_string(self):
        """Test that substring filter raises when only a string (not a tuple/list) is provided."""
        with self.assertRaises(ValueError):
            row_filter = SubstringFilter(substrings="exclude_me")

    def test_substring_filter_no_filter(self):
        row_filter = SubstringFilter(substrings=["nothing"])
        out = row_filter(self.df)
        pd.testing.assert_frame_equal(self.df, out)
