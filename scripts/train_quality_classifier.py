"""
Usage:
python scripts/train_quality_classifier.py --save_model --create_train_data
"""
import glob
import multiprocessing
import random

import fire
import nltk
import numpy as np
import pandas as pd
from tqdm import tqdm

nltk.download("punkt")

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from tabliblib.summarizers import TableSummarizer


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


def make_train_data():
    low_quality_files = glob.glob("/Users/jpgard/Documents/github/tablm/tmp/v6.0.0-sample/*.parquet")
    print(f"got {len(low_quality_files)} low-quality files")

    high_quality_fileglobs = [
        "/Users/jpgard/Documents/github/tablm/tmp/grinsztajn/**/*.csv",
        "/Users/jpgard/Documents/github/tablm/tmp/openml_cc18/**/*.csv",
        "/Users/jpgard/Documents/github/tablm/tmp/unipredict/**/*.csv",
        "/Users/jpgard/Documents/github/tablm/tmp/openml_ctr23/**/*.csv",
        "/Users/jpgard/Documents/github/tablm/tmp/ucidata/**/*.csv",
        "/Users/jpgard/Documents/github/tablm/tmp/pmlb/**/*.csv",
    ]
    high_quality_files = [f for fg in high_quality_fileglobs for f in glob.glob(fg, recursive=True)]

    all_files = low_quality_files + high_quality_files
    random.shuffle(all_files)

    num_cores = multiprocessing.cpu_count()

    # Create a pool of workers
    with multiprocessing.Pool(num_cores) as pool:
        # Use tqdm to show a progress bar
        all_metadata = list(tqdm(pool.imap(_process_file, all_files), total=len(all_files)))

    df = pd.DataFrame(all_metadata)
    df["quality"] = df["src_file"].apply(lambda f: int(f in low_quality_files))
    df.to_csv("table_quality_data.csv", index=False)
    return df


def main(create_train_data: bool = False,
         save_model: bool = True,
         filename: str = "table_quality_data.csv",
         model_path="xgb_quality_scorer.json",
         reload_for_eval: bool = False):
    if create_train_data:
        df = make_train_data()
    else:
        df = pd.read_csv(filename)

    df = df.replace({float("inf"): np.nan})

    # Summaries can contain very large values (too large for XGBoost); we clip them
    # to ensure they stay in a reasonable range.
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].clip(lower=-1e15, upper=1e15)

    target_colname = "quality"
    train_drop_colnames = ["src_file", "table_n"]  # table_n is the most predictive feature if allowed.
    train_df, test_df = train_test_split(df, train_size=0.8, random_state=42, stratify=df[target_colname])
    clf = XGBClassifier()
    clf.fit(train_df.drop(columns=[target_colname, *train_drop_colnames]), train_df[target_colname])

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
