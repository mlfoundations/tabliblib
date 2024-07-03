import io
import logging
from dataclasses import dataclass
from typing import Union, List

import numpy as np
import pandas as pd
import pyarrow as pa


def sample_columns_if_needed(df: pd.DataFrame, max_cols: int = 512) -> pd.DataFrame:
    """
    Checks if the dataframe has more than max_cols columns.
    If it does, returns a new dataframe with a random sample of max_cols columns.
    Otherwise, returns the original dataframe.

    Args:
    df (pandas.DataFrame): The original dataframe.
    max_cols (int, optional): Maximum number of columns allowed. Defaults to 512.

    Returns:
    pandas.DataFrame: The original dataframe or a sampled version of it.
    """
    if df.shape[1] > max_cols:  # More columns than allowed
        sampled_columns = np.random.choice(df.columns, size=max_cols, replace=False)
        return df[sampled_columns]
    return df


def write_arrow_bytes(df: pd.DataFrame) -> bytes:
    """Convert a pandas DataFrame to Arrow bytes."""
    table = pa.Table.from_pandas(df)
    buf = io.BytesIO()
    with pa.RecordBatchStreamWriter(buf, table.schema) as writer:
        writer.write_table(table)
    return buf.getvalue()


def arrow_to_pandas_safe(t: pa.Table, raise_on_error=False) -> Union[pd.DataFrame, None]:
    """Convert arrow tables to pandas dataframes, handling common edge cases."""
    if raise_on_error:
        return t.to_pandas()
    try:
        return t.to_pandas()
    except pa.lib.ArrowInvalid:
        return None
    except OSError as oe:
        if "Invalid IPC stream: negative continuation token" in str(oe):
            logging.warning(str(oe))
            return None
        else:
            raise oe
    except TypeError as te:
        # TODO(jpgard): consider filing a bug for this in pyarrow repo; should fall back to string or something instead of failing.
        if "data type 'time' not understood" in str(te):
            return None
        else:
            raise te
    except ValueError as ve:
        # ValueError happens due to non-unique indices or index levels,
        # which seem to be allowable in Arrow but not in Pandas.
        logging.warning(f"ValueError casting arrow table to pandas: {ve}")


def read_arrow_bytes(arrow_bytes: bytes, raise_on_error=False) -> Union[pd.DataFrame, None]:
    if raise_on_error:
        return arrow_to_pandas_safe(pa.RecordBatchStreamReader(arrow_bytes).read_all(), raise_on_error=False)
    try:
        return arrow_to_pandas_safe(pa.RecordBatchStreamReader(arrow_bytes).read_all())
    except pa.lib.ArrowInvalid:
        return None


@dataclass
class TabLibElement:
    """Container for a single element of TabLib."""
    job: str
    batch: str
    part: str
    key: str
    ref: str
    error: str
    bucket: str
    ref_id: str
    exec_id: str
    run_metadata: str
    context_metadata: str
    arrow_bytes_error: str
    arrow_bytes: bytes
    df: pd.DataFrame = None

    @property
    def content_hash(self) -> str:
        return self.key.split("/")[-1]

    @classmethod
    def from_series(cls, ser: pd.Series):
        return cls(**ser.to_dict())

    def get_df(self) -> Union[pd.DataFrame, None]:
        """Parse the arrow bytes if they have not been parsed already.

        This saves time parsing arrow bytes for tables we don't use
        (e.g. those thrown away by deduplication)."""
        if self.df is not None:
            return self.df
        else:
            if not self.arrow_bytes:
                return None
            self.df = read_arrow_bytes(self.arrow_bytes)
            return self.df


def load_shard(shard) -> List[TabLibElement]:
    assert shard.endswith(".parquet"), f"expected a parquet file; got {shard}"
    df = pd.read_parquet(shard, engine="fastparquet")
    # tmp = [pa.RecordBatchStreamReader(b).read_all() for b in df['arrow_bytes']]
    tmp = [TabLibElement.from_series(s) for _, s in df.iterrows()]
    return tmp
