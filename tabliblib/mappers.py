import hashlib
import logging
import sys
from typing import Dict, Any

import numpy as np

import tabliblib
from tabliblib.language_detection import detect


def add_content_hash(row: Dict[str, Any]) -> Dict[str, Any]:
    """Insert a content_hash column"""
    # This seems to be a base64 encoded hash.
    base64_hash = row["key"].split("/")[-1]
    row["content_hash"] = base64_hash

    hasher = hashlib.md5()
    hasher.update(base64_hash.encode("utf-8"))
    hx = hasher.digest()  # should be length 16
    row["sortkey1"] = np.array(int.from_bytes(hx[:8], byteorder=sys.byteorder), dtype=np.uint64)
    row["sortkey2"] = np.array(int.from_bytes(hx[8:], byteorder=sys.byteorder), dtype=np.uint64)

    return row


import pandas as pd


def drop_duplicates(batch: Dict[str, np.ndarray],
                    key="content_hash",
                    sort_by="path") -> Dict[str, np.ndarray]:
    inputs_len = len(batch[key])
    # Convert to pandas, drop duplicates, convert back to original format.
    # Sort by 'path' first,
    output = pd.DataFrame(batch).sort_values(sort_by).drop_duplicates(subset=[key]).to_dict("list")
    for k in output.keys():
        if k in ("sortkey1", "sortkey2"):
            output[k] = np.array(output[k], dtype=np.uint64)
        else:
            output[k] = np.array(output[k])
    logging.warning(f"got {inputs_len} inputs and dropped {inputs_len - len(output)} duplicates")
    return output


def add_dataframe_summary_info(row):
    """Parse the arrow bytes and add summary information about the dataset."""
    df = tabliblib.read_arrow_bytes(row["arrow_bytes"])
    if df is not None:
        row["nrows"], row["ncols"] = df.shape
        row["dtype_counts"] = str(df.dtypes.value_counts().to_dict())  # json.dumps() doesn't work with dtype keys
        # TODO(jpgard): play with formatting; this can make a big difference to langdetect.
        #  Must remove newlines for langdetect.
        headers = [str(c).replace("\n", "") for c in df.columns.tolist() if c is not None] if df is not None else []
        row["headers"] = " ".join(headers)
    else:
        row["nrows"] = None
        row["ncols"] = None
        row["dtype_counts"] = None
        row["headers"] = None
    return row


def detect_language(row):
    """See https://pypi.org/project/fasttext-langdetect/"""
    if row["headers"]:
        row["langdetect_result"] = detect(text=row["headers"], low_memory=False)
    else:
        row["langdetect_result"] = None
    return row
