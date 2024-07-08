"""
To run tests:
python -m unittest tabliblib/test_filters.py
"""
import random
import string
import unittest
from typing import List

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from tabliblib.summarizers import TableSummarizer
from tabliblib.filters import contains_pii, contains_code, compute_frac_null_like, is_kv_header, \
    compute_frac_contains_code, apply_row_based_filter, compute_frac_numeric_colnames


def generate_random_words() -> List[str]:
    return [''.join(random.choices(string.ascii_lowercase, k=random.randint(3, 10))) for _ in range(20)]


# Examples that should be detected as code
CODE_SAMPLES = [
    "for (int i = 0; i < 10; i++) { cout << i << endl; }",
    "def my_function(arg): return arg * 2",
    "if (x > 5) { return true; } else { return false; }",
    "var myVar = function(x) { return x * x; };",
    "console.log('Hello, world!');",
    "public class MyClass { int myMethod() { return 1; } }",
    "#include <iostream> int main() { std::cout << 'Hello World!'; return 0; }"
]

# Examples that should not be detected as code
NON_CODE_SAMPLES = [
    "This is a test sentence.",
    "This string contains for the word",
    # "I can finally use this string",
    "Here is a for loop, just kidding.",
    "Let's have a meeting at 10am.",
    "This string contains an = character."
    "We can float away.",
    "2024/01/05",
    "this.that.that.that.that",
    "A phrase (with a parenthetical) that is not code.",
    "This string contains https://www.google.com/ inside.",
    "A string containing index.html within.",
    "nan",
    "None",
    "none",
    "The quick brown fox jumps over the lazy dog."
]

PII_SAMPLES = [
    "+1 734 493-4903",
    "(734) 409-1203",
    "jpgard@cs.washington.edu",
    "jpgard@uw.edu",
    "my.email@gmail.com",
    "another.email@this.org",
]

NON_PII_SAMPLES = [
    "1234567",  # do not match general 7-digit numbers
    "1234567890",  # do not match general 10-digit numbers
    "text with @ symbol",
    "not@home",
]


class TestFilters(unittest.TestCase):
    def test_contains_pii(self):
        """Test that contains_pii() detects pii in text."""

        # Generate a series of random words, insert PII, and check both with and without spaces
        for sample in PII_SAMPLES:
            random_words = generate_random_words() + [sample]
            random.shuffle(random_words)
            to_test = ' '.join(random_words)
            self.assertTrue(contains_pii(to_test))

            if "@" in sample:
                # For email addresses, we can still detect them even if they are embedded in text
                # with no spaces. We cannot reliably detect phone numbers in the same context.
                to_test = ''.join(random_words)
                self.assertTrue(contains_pii(to_test))

    def test_not_contains_pii(self):
        """Test that contains_pii() does not match on 'pii-like' data."""

        for sample in NON_PII_SAMPLES:
            random_words = generate_random_words() + [sample]
            random.shuffle(random_words)
            to_test = ' '.join(random_words)
            self.assertFalse(contains_pii(to_test))

        # Check that function does not match on pii-like integer values.
        for to_test in [1234567, 1234567890]:
            self.assertFalse(contains_pii(to_test))

    def test_detects_code(self):
        """Test that code is properly detected."""
        for sample in CODE_SAMPLES:
            self.assertTrue(contains_code(sample), f"Failed to detect code: {sample}")

    def test_ignores_non_code(self):
        """Test that non-code is properly passed through."""
        for sample in NON_CODE_SAMPLES:
            self.assertFalse(contains_code(sample), f"Incorrectly detected non-code as code: {sample}")

    def test_compute_frac_contains_code(self):
        """Test that compute_frac_contains_code() works correctly."""
        self.assertEqual(compute_frac_contains_code(pd.Series(CODE_SAMPLES)), 1.)
        self.assertEqual(compute_frac_contains_code(pd.Series(NON_CODE_SAMPLES)), 0.)
        self.assertEqual(compute_frac_contains_code(pd.Series(CODE_SAMPLES[:3] + NON_CODE_SAMPLES[:3])), 0.5)
        self.assertEqual(compute_frac_contains_code(pd.Series(CODE_SAMPLES[:2] + NON_CODE_SAMPLES[:6])), 0.25)

    def test_compute_frac_null_like(self):
        test_ser = pd.Series([0.,
                              "this",
                              99,
                              "   some text   ",
                              False,
                              "",
                              "     ",
                              "\t",
                              np.nan,
                              None,
                              ])
        expected = 0.5
        actual = compute_frac_null_like(test_ser)
        self.assertEqual(expected, actual)

    def test_row_based_filter_numeric(self):
        """Test row-based filtering on numeric values."""
        df = pd.DataFrame({"x": [1, 2, 99], "y": [4, 5, 6]})
        filtered = apply_row_based_filter(df, filter_fn=lambda x: x > 10, string_columns_only=False)
        self.assertEqual(len(filtered), 2)

    def test_row_based_filter_pii(self):
        """Test row-based filtering on pii."""
        df = pd.DataFrame({"x": [1, 2, 3, 4],
                           "y": [NON_PII_SAMPLES[1], PII_SAMPLES[0], NON_PII_SAMPLES[2], PII_SAMPLES[1]]
                           })
        filtered = apply_row_based_filter(df, filter_fn=contains_pii, string_columns_only=True)
        self.assertEqual(len(filtered), 2)

    def test_row_based_filter_code(self):
        """Test row-based filtering on code."""
        df = pd.DataFrame({"x": [1, 2, 3, 4],
                           "y": [CODE_SAMPLES[1], CODE_SAMPLES[0], NON_CODE_SAMPLES[2], NON_CODE_SAMPLES[1]],
                           })
        filtered = apply_row_based_filter(df, filter_fn=contains_code, string_columns_only=True)
        self.assertEqual(len(filtered), 2)

    def test_row_based_filter_string_only(self):
        """Test that filter fn is applied only to string when string_columns_only is True."""
        df = pd.DataFrame({"x": [199, 299, 399], "y": ["string1", "string9", "string15"]})
        filtered = apply_row_based_filter(df, filter_fn=lambda x: "99" in str(x), string_columns_only=True)
        self.assertEqual(len(filtered), 3)

    def test_filter_rows_too_long(self):
        """Test that rows with cells containing too-long strings are filtered properly."""
        df = pd.DataFrame({"x": [199, 299, 399], "y": ["string1" * 100, "string9", "string15"]})
        filtered = apply_row_based_filter(df, filter_fn=lambda x: len(x) > 16, string_columns_only=True)
        self.assertEqual(len(filtered), 2)

    def test_max_frac_numeric_colnames(self):
        """Test that max_frac_numeric_colnames() filters tables properly."""
        df = pd.DataFrame({
            "x": [199, 299, 399],
            "y": [199, 299, 399],
            "01": [199, 299, 399],
            "02": [199, 299, 399],
            "03": [199, 299, 399],

        })
        self.assertEqual(compute_frac_numeric_colnames(df), 3 / 5.)

        df = pd.DataFrame({
            "x": [199, 299, 399],
            "1.456": [199, 299, 399],
            "-3.14": [199, 299, 399],
            "0000": [199, 299, 399],

        })
        self.assertEqual(compute_frac_numeric_colnames(df), 3 / 4.)

    def test_is_kv_header(self):
        """Test is_kv_header() detection of headers containing key-value pairs."""
        kv_headers = [
            '"key":"value"',
            '"key": "value"',
            '"key":"value"',
            '"key": "value"',
        ]
        for kv_header in kv_headers:
            self.assertTrue(is_kv_header(kv_header))

        non_kv_headers = [
            "some text that : happens to contain a colon",
            "text that contains 'this single quote' inside",
            'text that contains "this double quote" inside',
        ]
        for non_kv_header in non_kv_headers:
            self.assertFalse(is_kv_header(non_kv_header))


from tabliblib.filters import (TableFilterChain, RowCountFilter, ColumnCountFilter, BadHeadersFilter, SchemaFilter,
                               ValidColumnCountFilter, CodeDetectionFilter, PIIDetectionFilter,
                               TableQualityFilter)
from tabliblib.io import write_arrow_bytes


class TestTableFilterChain(unittest.TestCase):
    def test_simple_filter_chain(self):
        df = pd.DataFrame({
            "x": [199, 299, 399],
            "1.456": [199, 299, 399],
            "-3.14": [199, 299, 399],
            "0000": [199, 299, 399],

        })
        table_filter_chain = TableFilterChain([RowCountFilter(min_rows=2), ColumnCountFilter(min_columns=2)])
        self.assertTrue(table_filter_chain(df))

        table_filter_chain = TableFilterChain([RowCountFilter(min_rows=5), ColumnCountFilter(min_columns=2)])
        self.assertFalse(table_filter_chain(df))

    def test_simple_filter_chain_with_arrow_bytes(self):
        df = pd.DataFrame({
            "x": [199, 299, 399],
            "1.456": [199, 299, 399],
            "-3.14": [199, 299, 399],
            "0000": [199, 299, 399],

        })
        elem = {"arrow_bytes": write_arrow_bytes(df)}
        table_filter_chain = TableFilterChain([RowCountFilter(min_rows=2), ColumnCountFilter(min_columns=2)])
        self.assertTrue(table_filter_chain(elem))

        table_filter_chain = TableFilterChain([RowCountFilter(min_rows=5), ColumnCountFilter(min_columns=2)])
        self.assertFalse(table_filter_chain(elem))

    def test_simple_filter_chain_with_none(self):
        table_filter_chain = TableFilterChain([RowCountFilter(min_rows=5), ColumnCountFilter(min_columns=2)])
        self.assertFalse(table_filter_chain(None))

    def test_table_quality_filter(self, model_path="xgb_quality_scorer.json"):
        df = pd.DataFrame({
            "x": [199, 299, 399],
            "1.456": [199, 299, 399],
            "-3.14": [199, 299, 399],
            "0000": [199, 299, 399],

        })
        clf = XGBClassifier()
        summarizer = TableSummarizer()
        print(f"reloading model from saved checkpoint {model_path}")
        clf.load_model(model_path)
        quality_filter = TableQualityFilter(
            feature_extraction_fn=lambda x: pd.DataFrame([summarizer(x)]).drop(columns=["table_n"]),
            classifier=clf,
            threshold=1e-10)
        table_filter_chain = TableFilterChain([quality_filter])
        self.assertTrue(table_filter_chain(df))

    def test_long_filter_chain(self):
        df = pd.DataFrame({
            "x": [199, 299, 399],
            "1.456": [199, 299, 399],
            "-3.14": [199, 299, 399],
            "0000": [199, 299, 399],
            "category": ["A", "B", "C"],

        })
        table_filter_chain = TableFilterChain(
            [
                RowCountFilter(min_rows=2),
                ColumnCountFilter(min_columns=2),
                BadHeadersFilter(max_frac_numeric_colnames=0.75),
                SchemaFilter(min_dtypes=2),
                ValidColumnCountFilter(max_header_len_chars=100,
                                       min_unique_column_values=2,
                                       max_null_like_frac=1.,
                                       min_cols=2),
                CodeDetectionFilter(0.1),
                PIIDetectionFilter(0.1),
            ])
        self.assertTrue(table_filter_chain(df))

        # Same as above except with a valid column count filter that should reject the dataframe.
        table_filter_chain = TableFilterChain(
            [
                RowCountFilter(min_rows=2),
                ColumnCountFilter(min_columns=2),
                BadHeadersFilter(max_frac_numeric_colnames=0.75),
                SchemaFilter(min_dtypes=2),
                ValidColumnCountFilter(max_header_len_chars=100,
                                       min_unique_column_values=2,
                                       max_null_like_frac=1.,
                                       min_cols=6),  # <-- this should trigger rejection
                CodeDetectionFilter(0.1),
                PIIDetectionFilter(0.1),
            ])
        self.assertFalse(table_filter_chain(df))
