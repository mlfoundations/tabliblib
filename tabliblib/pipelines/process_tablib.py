"""
Reads a set of Tablib shards, apply a set of filters, and then write
individual tables out to files.

Usage:

Make sure to start the ray head node first with `ray start --head `. Then:

# local test
python scripts/process_tablib.py \
    --data_dir "sample-shards/tablib-v1-sample-tiny/" \
    --config_version v7 \
    --output_dir ./tmp/tablib_processed/v7v2/ \
    --read_mem_per_worker_gb 2 \
    --dedup_dir ./tmp/tablib_processed/dedup/


# run on hyak on full dataset
ray start --head --temp-dir=/gscratch/scrubbed/jpgard/ray-tmp
ray start --head --temp-dir=/gscratch/efml/jpgard/tabliblib/ray-tmp
unset http_proxy; unset https_proxy; \
python scripts/process_tablib.py \
    --data_dir "/data/tablib/tablib/" \
    --dedup_dir "/gscratch/efml/jpgard/tabliblib/dedup/content_hash_index/job4/" \
    --output_dir /gscratch/efml/jpgard/tabliblib/processed/v6.0.0-tmp/ \
    --config_version v6 \
    --chunk_size 512 \
    --chunk_index 0


NOTE: on SLURM, prefix the above command with `unset http_proxy; unset https_proxy;`.
"""
import glob
import logging
import os
import random
import time
from functools import partial
from typing import Optional, Dict, Any

import fire
import pandas as pd
import psutil
import pyarrow as pa
import ray
from xgboost import XGBClassifier

from tabliblib.config import PREPROCESS_VERSIONS, PreprocessConfig
from tabliblib.dataframe_utils import DataFrameFileDataSink
from tabliblib.dedup_utils import path_to_str
from tabliblib.filter.column_filters import ColumnFilterChain, MaxColumnsFilter, InvalidColumnsFilter
from tabliblib.filter.filter_utils import is_english
from tabliblib.filter.row_filters import RowFilterChain, MaxValueLengthFilter, SubstringFilter, CodeRegexFilter, \
    PIIRegexFilter, DuplicateRowsFilter, MaxRowCountFilter
from tabliblib.filter.table_filters import TableFilterChain, RowCountFilter, ColumnCountFilter, BadHeadersFilter, \
    TableQualityFilter, PIIDetectionFilter, CodeDetectionFilter, SchemaFilter, ValidColumnCountFilter
from tabliblib.io import read_arrow_bytes, write_arrow_bytes
from tabliblib.mappers import add_dataframe_summary_info, detect_language
from tabliblib.ray_utils import start_ray
from tabliblib.summarizers import TableSummarizer

RANDOM_SEED = 2974

BYTES_PER_GB = 1024 * 1024 * 1024


def make_table_filter_chain(preprocess_config: PreprocessConfig,
                            table_quality_filter: Optional[TableQualityFilter] = None) -> TableFilterChain:
    filter_chain = TableFilterChain([
        RowCountFilter(min_rows=preprocess_config.min_rows),
        ColumnCountFilter(min_columns=preprocess_config.min_cols,
                          max_columns=preprocess_config.max_cols if preprocess_config.filter_too_many_columns else None),
        BadHeadersFilter(
            max_frac_numeric_colnames=preprocess_config.max_frac_numeric_colnames,
            max_frac_unnamed_columns=preprocess_config.max_frac_unnamed_columns),
        SchemaFilter(preprocess_config.min_dtypes),
        ValidColumnCountFilter(
            max_header_len_chars=preprocess_config.max_header_len_chars,
            min_unique_column_values=preprocess_config.min_unique_column_values,
            max_null_like_frac=preprocess_config.max_null_like_frac,
            min_cols=preprocess_config.min_cols),
        CodeDetectionFilter(preprocess_config.code_detect_filter_threshold),
        PIIDetectionFilter(preprocess_config.pii_detect_filter_threshold),

    ])
    if preprocess_config.table_quality_classifier_position == "pre":
        assert table_quality_filter is not None
        filter_chain.append(table_quality_filter)
    return filter_chain


def make_column_filter_chain(preprocess_config: PreprocessConfig) -> ColumnFilterChain:
    column_filter_chain = ColumnFilterChain()
    if preprocess_config.drop_invalid_cols:
        column_filter_chain.append(
            InvalidColumnsFilter(
                max_header_len_chars=preprocess_config.max_header_len_chars,
                min_unique_column_values=preprocess_config.min_unique_column_values,
                max_null_like_frac=preprocess_config.max_null_like_frac
            )
        )
    if preprocess_config.drop_extra_cols:
        column_filter_chain.append(
            MaxColumnsFilter(preprocess_config.max_cols)
        )
    return column_filter_chain


def make_row_filter_chain(preprocess_config: PreprocessConfig) -> RowFilterChain:
    row_filter_chain = RowFilterChain()
    if preprocess_config.max_value_len_chars:
        row_filter_chain.append(MaxValueLengthFilter(preprocess_config.max_value_len_chars))
    if preprocess_config.filter_rows_containing_substrings:
        row_filter_chain.append(SubstringFilter(preprocess_config.filter_rows_containing_substrings))
    if preprocess_config.filter_rows_containing_code:
        row_filter_chain.append(CodeRegexFilter())
    if preprocess_config.filter_rows_containing_pii:
        row_filter_chain.append(PIIRegexFilter())
    if preprocess_config.drop_duplicate_rows:
        row_filter_chain.append(DuplicateRowsFilter())
    if preprocess_config.drop_extra_rows:
        row_filter_chain.append(MaxRowCountFilter(preprocess_config.max_output_rows))
    return row_filter_chain


def get_parallelism(num_cores, available_memory, partition_size):
    max_partitions_by_cores = num_cores
    max_partitions_by_memory = available_memory // partition_size
    return min(max_partitions_by_cores, max_partitions_by_memory)


def main(
        config_version: str,
        dedup_dir: str,
        data_dir: str = "./sample-shards/",
        chunk_size: int = 512,
        chunk_index: int = 0,
        shuffle_input_files: bool = True,
        output_dir: str = "./processed",
        force_parallelism: Optional[int] = None,
        read_mem_per_worker_gb: int = 16,
        write_mem_per_worker_gb: int = 32,
):
    chunk_index = int(chunk_index)
    chunk_size = int(chunk_size)

    assert config_version in PREPROCESS_VERSIONS.keys(), \
        f"invalid config version {config_version}; must be one of {PREPROCESS_VERSIONS.keys()}"
    data_dir = os.path.abspath(data_dir)
    output_dir = os.path.abspath(output_dir)
    preprocess_config = PREPROCESS_VERSIONS[config_version]

    print(f"[DEBUG] data_dir is {data_dir}")

    assert os.path.exists(data_dir)
    fileglob = os.path.join(os.path.abspath(data_dir), "**", "*.parquet")
    print(f"listing files matching {fileglob}")
    files = glob.glob(fileglob, recursive=True)
    print(f"got {len(files)} files matching {fileglob}")
    if not len(files):
        return

    if shuffle_input_files:  # useful in combination with max_shards to get a random sample.
        random.seed(RANDOM_SEED)
        random.shuffle(files)

    chunk_start = chunk_index * chunk_size
    print(f"fetching chunk number {chunk_index} of size {chunk_size} from {len(files)} input files")
    files = files[chunk_start:chunk_start + chunk_size]

    output_dir = os.path.join(output_dir, f"chunk-{chunk_index:04}")
    print(f"output_dir is {output_dir}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    start_ray()

    num_nodes = len(ray.nodes())
    print(f"num nodes = {num_nodes}")
    num_cores = os.cpu_count()
    print(f"num cores = {num_cores}")
    if force_parallelism is None:
        # parallelism = num_nodes * num_cores
        # Example usage
        num_cores = os.cpu_count()  # Total number of cores across all workers
        available_memory = psutil.virtual_memory().available  # Available memory in bytes
        partition_size = BYTES_PER_GB  # Desired partition size (e.g., 1 GB)
        parallelism = get_parallelism(num_cores, available_memory, partition_size)

    else:
        parallelism = force_parallelism
    print(f"parallelism = {parallelism}")

    ctx = ray.data.DataContext.get_current()
    ctx.execution_options.verbose_progress = True
    ctx.max_errored_blocks = 1000

    start = time.time()

    def read_deduped(row: Dict[str, str]):
        """Efficiently read deduped data from a single shard."""
        shard_path = row['item']  # this is a path to a tablib shard
        shard_path_str = path_to_str(shard_path)
        fileglob = os.path.join(dedup_dir, f"*{shard_path_str}*.parquet")
        files = glob.glob(fileglob)

        if not len(files):
            # Shards containing only duplicates do exist (but are rare); handle this case.
            logging.warning(f"no files matching {fileglob} for path {shard_path} with shard_path_str {shard_path_str}")
            return iter([])

        # Note: reading + concatenating the parquet files here is extremely fast (~0.2s total)
        # relative to reading the pyarrow table further below (~40s)
        logging.debug(f"got {len(files)} files for shard_path_str {shard_path_str}")
        start = time.time()
        df = pd.concat(pd.read_parquet(f) for f in files)
        logging.debug(
            f"finished reading {len(files)} files for shard_path_str {shard_path_str} in {time.time() - start} secs")
        keys_to_keep = set(df["key"].unique().tolist())

        start = time.time()
        # Create a PyArrow dataset from the Parquet file
        dataset = pa.dataset.dataset(shard_path, format="parquet")
        # Create a filter expression to keep only the desired keys
        filter_expression = pa.dataset.field("key").isin(keys_to_keep)

        # Apply the filter expression to the dataset and select the columns
        table = dataset.filter(filter_expression).to_table(columns=["key", "arrow_bytes"])

        # Add a new column by mapping a function over the table
        new_column = pa.array(list(map(lambda row: row["key"].split("/")[-1], table.to_pylist())))
        table = table.append_column("content_hash", new_column)
        logging.warning(f"finished reading {shard_path} in {time.time() - start} secs")

        # Iterate over the rows of the filtered table and yield each row
        for row in table.to_pylist():
            yield row

    # .from_items() creates schema {'item': item}
    ray_remote_args = {"num_cpus": 1}
    if os.cpu_count() > 4:
        ray_remote_args.update({"memory": read_mem_per_worker_gb * BYTES_PER_GB})
    print(f"ray_remote_args is {ray_remote_args}")
    ds = ray.data.from_items(files).flat_map(read_deduped, **ray_remote_args)

    print(f"[INFO] finished reading files {files}")
    print(f"[INFO] schema is {ds.schema()}")
    print(f"[INFO] ds is {ds}")

    ds = ds.map(add_dataframe_summary_info) \
        .map(detect_language)

    if preprocess_config.table_quality_classifier:
        summarizer = TableSummarizer()
        clf = XGBClassifier()
        print(f"reloading model from saved checkpoint {preprocess_config.table_quality_classifier}")
        clf.load_model(preprocess_config.table_quality_classifier)
        table_quality_filter = TableQualityFilter(
            feature_extraction_fn=lambda x: pd.DataFrame([summarizer(x)]).drop(columns=["table_n"]),
            classifier=clf,
            threshold=preprocess_config.table_quality_threshold)
    else:
        table_quality_filter = None

    table_filter_chain_pre = make_table_filter_chain(preprocess_config, table_quality_filter=table_quality_filter)

    table_filter_chain_post = TableFilterChain([
        RowCountFilter(min_rows=preprocess_config.min_rows)])
    if preprocess_config.table_quality_classifier_position == "post":
        table_filter_chain_post.append(table_quality_filter)

    column_filter_chain = make_column_filter_chain(preprocess_config)

    row_filter_chain = make_row_filter_chain(preprocess_config)

    def _column_filter_map_fn(row: Dict[str, Any]):
        df = read_arrow_bytes(row["arrow_bytes"], raise_on_error=True)
        df_out = column_filter_chain(df)
        if df_out is not None and len(df_out):
            row["arrow_bytes"] = write_arrow_bytes(df_out)
        else:
            row["arrow_bytes"] = None
        return row

    def _row_filter_map_fn(row: Dict[str, Any]):
        df = read_arrow_bytes(row["arrow_bytes"], raise_on_error=True)
        df_out = row_filter_chain(df)
        if df_out is not None and len(df_out):
            row["arrow_bytes"] = write_arrow_bytes(df_out)
        else:
            row["arrow_bytes"] = None
        return row

    _english_filter = partial(is_english, threshold=preprocess_config.langdetect_threshold)
    ds = (ds
          .filter(_english_filter)
          .filter(table_filter_chain_pre)
          .map(_column_filter_map_fn)
          .map(_row_filter_map_fn)
          .filter(table_filter_chain_post))

    # Allocate more resources to writing; this requires more memory bc arrow bytes are expanded
    #  into a pandas dataframe.
    data_frame_sink = DataFrameFileDataSink(
        output_dir,
        output_format="parquet",
        mem_per_writer=write_mem_per_worker_gb if write_mem_per_worker_gb else 2 * read_mem_per_worker_gb,
    )
    # Write the dataset to CSV files
    result = data_frame_sink.write(ds)

    # Important!! Fetch and display the paths of the written CSV files. Do not remove this;
    # it materializes the dataset which ensures that the files are actually written
    output = result.take_all()
    print(f"processed {len(output)} outputs.")

    end = time.time()
    print(f"execution finished in {int(end - start)}s.")


if __name__ == "__main__":
    fire.Fire(main)
