"""
To run tests:
python -m unittest tabliblib/test_io.py -v
"""
import unittest
from tabliblib.io import read_arrow_bytes, write_arrow_bytes
import pandas as pd

class TestArrowBytesRoundTrip(unittest.TestCase):
    """Test that write -> read round-trip recorvers original dataframe."""
    def test_write_and_read_bytes(self):
        df_original = pd.DataFrame({
            'int_col': [1, 2, 3],
            'float_col': [1.1, 2.2, 3.3],
            'str_col': ['a', 'b', 'c']
        })

        # Write DataFrame to Arrow bytes
        arrow_bytes = write_arrow_bytes(df_original)

        # Read Arrow bytes back to DataFrame
        df_roundtrip = read_arrow_bytes(arrow_bytes)

        # Check if the roundtrip DataFrame is equal to the original
        pd.testing.assert_frame_equal(df_original, df_roundtrip)
        print("Test passed: The roundtrip DataFrame is equal to the original.")
