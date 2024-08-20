import glob
import logging
import multiprocessing
from pprint import pprint
from typing import Sequence, Dict

import fire
import numpy as np
import pandas as pd
from tqdm import tqdm
from xgboost import XGBClassifier

from tabliblib.summarizers import TableSummarizer

summarizer = TableSummarizer()

from pyarrow.lib import ArrowInvalid


def _process_file(f):
    if f.endswith(".parquet"):
        try:
            df = pd.read_parquet(f)
        except ArrowInvalid:
            logging.warning(f"Skipping invalid parquet file {f}")
            return None
    elif f.endswith(".csv"):
        df = pd.read_csv(f)
    else:
        raise ValueError(f"got unexpected file type: {f}")

    # file_metadata = extract_quality_scoring_metadata(df)
    file_metadata = summarizer(df)
    file_metadata["src_file"] = f
    return file_metadata


def prepare_data(low_quality_fileglob: str):
    all_files = glob.glob(low_quality_fileglob)[:200]

    num_cores = multiprocessing.cpu_count()

    # Create a pool of workers
    with multiprocessing.Pool(num_cores) as pool:
        # Use tqdm to show a progress bar
        all_metadata = list(tqdm(pool.imap(_process_file, all_files), total=len(all_files)))
    # drop null results
    all_metadata = [x for x in all_metadata if x is not None]
    df = pd.DataFrame(all_metadata)
    return df


def compute_quantiles(clf,
                      df,
                      quantiles: Sequence[float] = (0.5, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99, 0.999)) -> Dict[
    float, float]:
    pred_probs = clf.predict_proba(df[clf.get_booster().feature_names])

    out: Dict[float, float] = {}
    # Assuming pred_probs[:,1] is your data
    data = pred_probs[:, 1]
    for q in quantiles:
        out[q] = np.quantile(data, q)
    return out


def main(
        trained_model_json: str,
        low_quality_fileglob: str = "../../tablm/tmp/v6.0.0-sample-valid/*.parquet",
):
    df = prepare_data(low_quality_fileglob)
    clf = XGBClassifier()

    clf.load_model(trained_model_json)
    quantiles = compute_quantiles(clf, df)
    pprint(quantiles)


if __name__ == "__main__":
    fire.Fire(main)
