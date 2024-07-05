"""
Reads a set of Tablib shards, apply a set of filters, and then write
individual tables out to files.

Usage:

Make sure to start the ray head node first with `ray start --head `. Then:

# local test, with dedup
python scripts/process_tablib.py \
    --data_dir "sample-shards/tablib-v1-sample-tiny/" \
    --output_dir ./tmp/tablib_processed/v1-sample-tiny/ \
    --read_mem_per_worker_gb 2 \
    --dedup_dir ./tmp/tablib_processed/dedup/ \
    --config_version v6


# run on hyak on full dataset, deduped
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
from typing import Optional, Dict

import fire
import pandas as pd
import psutil
import pyarrow as pa
import ray

from tabliblib import filters
from tabliblib.config import PREPROCESS_VERSIONS
from tabliblib.dataframe_utils import DataFrameFileDataSink
from tabliblib.dedup_utils import path_to_str
from tabliblib.filters import is_english
from tabliblib.mappers import add_dataframe_summary_info, detect_language
from tabliblib.ray_utils import start_ray

RANDOM_SEED = 2974

BYTES_PER_GB = 1024 * 1024 * 1024


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

    table_filter_chain = filters.TableFilterChain([
        filters.RowCountFilter(min_rows=preprocess_config.min_rows),
        filters.ColumnCountFilter(min_columns=preprocess_config.min_cols,
                                  max_columns=preprocess_config.max_cols if preprocess_config.filter_too_many_columns else None),
        filters.BadHeadersFilter(max_frac_numeric_colnames=preprocess_config.max_frac_numeric_colnames,
                                 max_frac_unnamed_columns=preprocess_config.max_frac_unnamed_columns),
        filters.SchemaFilter(preprocess_config.min_dtypes),
        filters.ValidColumnCountFilter(max_header_len_chars=preprocess_config.max_header_len_chars,
                                       min_unique_column_values=preprocess_config.min_unique_column_values,
                                       max_null_like_frac=preprocess_config.max_null_like_frac,
                                       min_cols=preprocess_config.min_cols),
        filters.CodeDetectionFilter(preprocess_config.code_detect_filter_threshold),
        filters.PIIDetectionFilter(preprocess_config.pii_detect_filter_threshold),

    ])

    # TODO(jpgard): do language detection inside the dataframe filter fn, so we only have to parse the
    #  dataframe one time.
    _english_filter = partial(is_english, threshold=preprocess_config.langdetect_threshold)
    ds = ds \
        .filter(_english_filter) \
        .filter(table_filter_chain)

    # Allocate more resources to writing; this requires more memory bc arrow bytes are expanded
    #  into a pandas dataframe.
    data_frame_sink = DataFrameFileDataSink(
        output_dir,
        output_format="parquet",
        config=preprocess_config,
        mem_per_writer=write_mem_per_worker_gb if write_mem_per_worker_gb else 2 * read_mem_per_worker_gb)
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
