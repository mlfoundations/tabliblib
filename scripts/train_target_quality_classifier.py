"""

Note: it is recommended to run the following to clean up tine_sklearn results after this script:
rm -rf ~/ray_results/
"""
import multiprocessing as mp
import os
import re
import time
import uuid
from typing import List, Any, Sequence

import fire
import numpy as np
import pandas as pd
import yaml
from rtfm.tree_baselines import tune_xgb
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
from tqdm import tqdm
# Note: it is important to import this early. Importing later
# seems to cause segmentation faults, potentially related to libomp see discussion here:
# https://github.com/dmlc/xgboost/issues/7039#issuecomment-860910066
from xgboost import XGBClassifier

from tabliblib.summarizers import SingleColumnSummarizer

# TODO(jpgard): should we remove DateTime summarizer? Verify that there
#  exist DateTime columns in the target datasets.
summarizer = SingleColumnSummarizer(agg_fns={},
                                    agg_quantiles=[],
                                    include_table_summary_metrics=False)


def process_files(root, dirs, files) -> List[Any]:
    obs = []
    # Skip UCI "variables" subdirectories
    if any([re.match("variables.*\\.csv", f) for f in files]):
        return []

    csv_file = [f for f in files if f.endswith('.csv')]
    if not csv_file:
        # Case: this is probably a higher-level directory that does not
        # contain any data itself; skipt it.
        return []

    jsonl_file = [f for f in files if f.endswith('.jsonl')]
    if not jsonl_file:
        print(f"skipping root {root} due to no jsonl file")
        return []

    if csv_file:
        # every directory with a csv file should contain a FeatureList jsonl.
        df = pd.read_csv(os.path.join(root, csv_file[0]))
        feature_list = pd.read_json(os.path.join(root, jsonl_file[0]), lines=True, dtype=object)
        feature_list["name"] = feature_list["name"].astype(str)
        if not feature_list.is_target.sum() == 1:
            raise ValueError(f"Expected one target for root {root}.")
        target_colname = feature_list[feature_list.is_target == True]["name"].item()
        if not target_colname in df.columns:
            raise ValueError(
                f"target column from FeatureList {target_colname} not in data columns {sorted(df.columns)}")
        for colname in df.columns:
            col_sum = summarizer(df[colname])
            is_target = feature_list.loc[feature_list.name == colname, "is_target"].item()
            col_sum["target"] = int(is_target)
            obs.append(col_sum)
    return obs


def main(save_model: bool = True,
         reload_for_eval: bool = False,
         output_dir: str = "target_quality_clf",
         n_trials: int = 50,
         train_data_dirs: Sequence[str] = (
                 "../../tablm/tmp/grinsztajn/",
                 "../../tablm/tmp/openml_cc18/",
                 "../../tablm/tmp/unipredict/",
                 "../../tablm/tmp/openml_ctr23/",
                 "../../tablm/tmp/pmlb/",
                 "../../tablm/tmp/ucidata/",
         )
         ):
    print("#" * 50)
    if isinstance(train_data_dirs, str):
        train_data_dirs = (train_data_dirs,)

    # Generate a unique run ID
    run_id = str(uuid.uuid4())
    print(f"run_id for this run is {run_id}")

    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.abspath(os.path.join(output_dir, f"xgb_target_quality_scorer_{run_id}.json"))

    # Save kwargs and run_id to a YAML file
    kwargs = {
        "output_dir": output_dir,
        "save_model": save_model,
        "model_path": model_path,
        "reload_for_eval": reload_for_eval,
        "train_data_dirs": train_data_dirs,
        "run_id": run_id,
        "n_trials": n_trials,
    }

    yaml_file = os.path.join(output_dir, f"run_config_{run_id}.yaml")
    print(f"writing kwargs to file {yaml_file}: {kwargs}")

    with open(yaml_file, 'w') as yaml_file:
        yaml.dump(kwargs, yaml_file)

    total_dirs = sum(len(list(os.walk(d))) for d in train_data_dirs)

    with tqdm(total=total_dirs, unit='dir', desc='Processing', ncols=70) as pbar:
        start_time = time.time()
        processed_dirs = 0

        obs: List[pd.Series] = []
        with mp.Pool(processes=mp.cpu_count()) as pool:
            for directory in train_data_dirs:
                for root, subdirs, files in os.walk(directory):
                    # obs.extend(process_files(root, subdirs, files))
                    obs.extend(pool.apply_async(process_files, args=(root, subdirs, files)).get())
                    processed_dirs += 1
                    pbar.update(1)

                    # Update estimated time
                    elapsed_time = time.time() - start_time
                    estimated_total_time = (elapsed_time / processed_dirs) * total_dirs
                    estimated_remaining_time = estimated_total_time - elapsed_time
                    pbar.set_postfix({'ETA': f'{estimated_remaining_time:.2f}s'})

    df = pd.DataFrame([x for x in obs if x is not None])
    # Summaries can contain very large values (too large for XGBoost); we clip them
    # to ensure they stay in a reasonable range.
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].clip(lower=-1e15, upper=1e15)

    target_colname = "target"
    train_df, test_df = train_test_split(df, train_size=0.8, random_state=42, stratify=df[target_colname])

    clf = tune_xgb(X_tr=train_df.drop(columns=[target_colname]),
                   y_tr=train_df[target_colname],
                   n_trials=n_trials)
    clf = clf.best_estimator_
    assert isinstance(clf, XGBClassifier)

    if save_model:
        print(f"saving model to {model_path}")
        clf.save_model(model_path)
        if reload_for_eval:
            print(f"reloading model from saved checkpoint {model_path}")
            clf.load_model(model_path)
    else:
        assert not reload_for_eval, "reload_for_eval requires save_model to be set to True."

    preds = clf.predict(test_df.drop(columns=[target_colname]))
    pred_probs = clf.predict_proba(test_df.drop(columns=[target_colname]))
    acc = accuracy_score(test_df[target_colname], preds)
    auc = roc_auc_score(test_df[target_colname], pred_probs[:, 1])
    print(f"accuracy is {acc:.4f}, auc is {auc:.4f}")

    majority_acc = max(
        test_df[target_colname].mean().item(),
        1 - test_df[target_colname].mean().item()
    )
    print(f"majority class accuracy is {majority_acc:.4f}")

    report = classification_report(y_true=test_df[target_colname], y_pred=preds,
                                   target_names=["non_target", "is_target"],
                                   digits=3)
    print(report)

    imp_df = (pd.DataFrame({"feature": train_df.drop(columns=[target_colname]).columns,
                            "importance": clf.feature_importances_})
              .sort_values("importance", ascending=False)
              .reset_index(drop=True)
              )

    print(imp_df.head(20))

    return


if __name__ == "__main__":
    fire.Fire(main)
