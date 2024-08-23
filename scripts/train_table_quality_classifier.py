"""
Usage:
python scripts/train_quality_classifier.py --save_model --create_train_data
"""
import glob
import multiprocessing
import os
import random
import uuid
from typing import Sequence, Optional, Union

import fire
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from tabliblib.summarizers import TableSummarizer
from rtfm.tree_baselines import tune_xgb


def _process_file(f):
    if f.endswith(".parquet"):
        df = pd.read_parquet(f)
    elif f.endswith(".csv"):
        df = pd.read_csv(f)
    else:
        raise ValueError(f"got unexpected file type: {f}")

    # file_metadata = extract_quality_scoring_metadata(df)
    summarizer = TableSummarizer()
    file_metadata = summarizer(df)
    file_metadata["src_file"] = f
    return file_metadata


def make_train_data(high_quality_fileglobs: Union[str, Sequence[str]],
                    low_quality_fileglob: str) -> pd.DataFrame:
    if isinstance(high_quality_fileglobs, str):
        high_quality_fileglobs = [high_quality_fileglobs, ]

    low_quality_files = glob.glob(low_quality_fileglob)
    print(f"got {len(low_quality_files)} low-quality files matching {low_quality_fileglob}")

    high_quality_files = [f for fg in high_quality_fileglobs for f in glob.glob(fg, recursive=True)]
    print(f"got {len(high_quality_files)} high-quality files matching {high_quality_fileglobs}")

    all_files = low_quality_files + high_quality_files
    random.shuffle(all_files)

    num_cores = multiprocessing.cpu_count()

    # Create a pool of workers
    with multiprocessing.Pool(num_cores) as pool:
        # Use tqdm to show a progress bar
        all_metadata = list(tqdm(pool.imap(_process_file, all_files), total=len(all_files)))

    df = pd.DataFrame(all_metadata)
    df["quality"] = df["src_file"].apply(lambda f: int(f in high_quality_files))
    return df


def main(create_train_data: bool = True,
         save_train_data: bool = True,
         save_model: bool = True,
         train_data_csv: Optional[str] = None,
         reload_for_eval: bool = False,
         high_quality_fileglobs: Sequence[str] = (
                 "../../tablm/tmp/grinsztajn/**/*.csv",
                 "../../tablm/tmp/openml_cc18/**/*.csv",
                 "../../tablm/tmp/unipredict/**/*.csv",
                 "../../tablm/tmp/openml_ctr23/**/*.csv",
                 "../../tablm/tmp/ucidata/**/*.csv",
                 "../../tablm/tmp/pmlb/**/*.csv",
         ),
         low_quality_fileglob: str = "../../tablm/tmp/v6.0.0-sample/*.parquet",
         output_dir: str = "table_quality_clf",
         n_trials: int = 20,
         ):
    print("#" * 50)

    # Generate a unique run ID
    run_id = str(uuid.uuid4())
    print(f"run_id for this run is {run_id}")

    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.abspath(os.path.join(output_dir, f"xgb_table_quality_scorer_{run_id}.json"))

    if not train_data_csv:
        train_data_csv = os.path.abspath(os.path.join(output_dir, f"table_quality_data_{run_id}.csv"))

    # Save kwargs and run_id to a YAML file
    kwargs = {
        "create_train_data": create_train_data,
        "save_train_data": save_train_data,
        "output_dir": output_dir,
        "save_model": save_model,
        "train_data_csv": train_data_csv,
        "model_path": model_path,
        "reload_for_eval": reload_for_eval,
        "high_quality_fileglobs": high_quality_fileglobs,
        "low_quality_fileglob": low_quality_fileglob,
        "run_id": run_id,
        "n_trials": n_trials,
    }
    yaml_file = os.path.join(output_dir, f"run_config_{run_id}.yaml")
    print(f"writing kwargs to file {yaml_file}: {kwargs}")

    with open(yaml_file, 'w') as yaml_file:
        yaml.dump(kwargs, yaml_file)

    if create_train_data:
        df = make_train_data(high_quality_fileglobs=high_quality_fileglobs,
                             low_quality_fileglob=low_quality_fileglob)
        if save_train_data:
            df.to_csv(train_data_csv, index=False)
    else:
        df = pd.read_csv(train_data_csv)

    df = df.replace({float("inf"): np.nan})

    # Summaries can contain very large values (too large for XGBoost); we clip them
    # to ensure they stay in a reasonable range.
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].clip(lower=-1e15, upper=1e15)

    target_colname = "quality"
    train_drop_colnames = ["src_file", "table_n"]  # table_n is the most predictive feature if allowed.
    train_df, test_df = train_test_split(df, train_size=0.8, random_state=42, stratify=df[target_colname])

    clf = tune_xgb(X_tr=train_df.drop(columns=[target_colname, *train_drop_colnames]),
                   y_tr=train_df[target_colname],
                   n_trials=n_trials)
    clf = clf.best_estimator_

    if save_model:
        print(f"saving model to {model_path}")
        clf.save_model(model_path)
        if reload_for_eval:
            print(f"reloading model from saved checkpoint {model_path}")
            clf.load_model(model_path)
    else:
        assert not reload_for_eval, "reload_for_eval requires save_model to be set to True."

    preds = clf.predict(test_df.drop(columns=[target_colname, *train_drop_colnames]))
    pred_probs = clf.predict_proba(test_df.drop(columns=[target_colname, *train_drop_colnames]))
    acc = accuracy_score(test_df[target_colname], preds)
    auc = roc_auc_score(test_df[target_colname], pred_probs[:, 1])
    print(f"accuracy is {acc:.4f}, auc is {auc:.4f}")

    majority_acc = max(
        test_df['quality'].mean().item(),
        1 - test_df['quality'].mean().item()
    )
    print(f"majority class accuracy is {majority_acc:.4f}")

    report = classification_report(y_true=test_df[target_colname], y_pred=preds,
                                   target_names=["low_quality", "high_quality"],
                                   digits=3)
    print(report)

    imp_df = (pd.DataFrame({"feature": train_df.drop(columns=[target_colname, *train_drop_colnames]).columns,
                            "importance": clf.feature_importances_})
              .sort_values("importance", ascending=False)
              .reset_index(drop=True)
              )

    print(imp_df.head(20))


if __name__ == "__main__":
    fire.Fire(main)
