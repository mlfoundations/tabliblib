"""
Tests for row filters.

To run tests:
python -m unittest tabliblib/test_row_filters.py -v
"""
import unittest

import pandas as pd

from tabliblib.filter.row_filters import SubstringFilter, MaxValueLengthFilter, CodeRegexFilter, PIIRegexFilter, \
    DuplicateRowsFilter
from tabliblib.test_filters import CODE_SAMPLES, NON_CODE_SAMPLES, PII_SAMPLES, NON_PII_SAMPLES


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


class TestCodeRegexFilter(unittest.TestCase):

    def test_code_regex_filter(self):
        df = pd.DataFrame(
            {"x": [1, 2, 3, 4],
             "y": [CODE_SAMPLES[1], CODE_SAMPLES[0], NON_CODE_SAMPLES[2], NON_CODE_SAMPLES[1]],
             })
        row_filter = CodeRegexFilter()
        out = row_filter(df)
        pd.testing.assert_frame_equal(out, df.iloc[2:])

    def test_code_regex_nofilter(self):
        """Test code regex filter does not exclude rows with no code."""
        df = pd.DataFrame(
            {"x": [1, 2, 3, 4],
             "y": [NON_CODE_SAMPLES[1], NON_CODE_SAMPLES[0], NON_CODE_SAMPLES[2], NON_CODE_SAMPLES[1]],
             })
        row_filter = CodeRegexFilter()
        out = row_filter(df)
        pd.testing.assert_frame_equal(out, df)


class TestPIIRegexFilter(unittest.TestCase):
    def test_pii_regex_filter(self):
        df = pd.DataFrame({
            "x": [1, 2, 3, 4, 5],
            "y": [NON_PII_SAMPLES[1], NON_PII_SAMPLES[2], PII_SAMPLES[0], PII_SAMPLES[1], PII_SAMPLES[2]]
        })
        row_filter = PIIRegexFilter()
        out = row_filter(df)
        pd.testing.assert_frame_equal(out, df[:2])

    def test_pii_regex_nofilter(self):
        df = pd.DataFrame({
            "x": [1, 2, 3, 4, 5],
            "y": [NON_PII_SAMPLES[1], NON_PII_SAMPLES[2], NON_PII_SAMPLES[3], NON_PII_SAMPLES[4], NON_PII_SAMPLES[5]]
        })
        row_filter = PIIRegexFilter()
        out = row_filter(df)
        pd.testing.assert_frame_equal(out, df)


class TestDuplicateRowsFilter(unittest.TestCase):
    def test_duplicate_rows_filter(self):
        df = pd.DataFrame({
            "x": [1, 2, 3, 4, 5, 5, 5],
            "y": ["a", "b", "c", "d", "e", "e", "e"]
        })
        row_filter = DuplicateRowsFilter()
        out = row_filter(df)
        pd.testing.assert_frame_equal(out, df.iloc[:5])

    def test_duplicate_rows_nofilter(self):
        """Test duplicate row filter does not modify DataFrame with no duplicates."""
        df = pd.DataFrame({
            "x": [1, 2, 3, 4, 5, ],
            "y": ["a", "b", "c", "d", "e", ]
        })
        row_filter = DuplicateRowsFilter()
        out = row_filter(df)
        pd.testing.assert_frame_equal(out, df)
