"""
Build a table of which files contain which content hashes.
# local test
python scripts/dedup_tablib.py \
    --data_dir "sample-shards/tablib-v1-sample-tiny/" \
    --max_shards 1 \
    --read_mem_per_worker_gb 1 \
    --output_dir ./tmp/tablib_processed/dedup/

# local test
python scripts/dedup_tablib.py \
    --data_dir "sample-shards/tablib-v1-sample" \
    --max_shards 3 \
    --read_mem_per_worker_gb 1 \
    --output_dir ./tmp/tablib_processed/dedup/

# process full tablib dataset on hyak
ray start --head --temp-dir=/gscratch/scrubbed/jpgard/ray-tmp
unset http_proxy; unset https_proxy; \
    python scripts/dedup_tablib.py \
    --data_dir "/data/tablib/tablib/" \
    --output_dir "/gscratch/efml/jpgard/tabliblib/dedup/content_hash_index/job4/" \
    --force-parallelism 1000
"""
import glob
import hashlib
import logging
import os
import time
from typing import Optional, Dict, Any, Sequence

import fire
import numpy as np
import pandas as pd
import psutil
import ray

from tabliblib.mappers import add_content_hash, drop_duplicates
from tabliblib.ray_utils import start_ray
from tabliblib.dedup_utils import path_to_str

RANDOM_SEED = 2974
BYTES_PER_GB = 1024 * 1024 * 1024


def write_file_output(batch: Dict[str, np.ndarray], output_dir: str) -> Dict[str, Sequence[str]]:
    """Write results to a separate parquet file for each input file."""
    df = pd.DataFrame(batch)
    outfiles = []

    # path_df contains all unique tables after deduplicating that are contained in the input
    # file indicated by path; note that due to batching there might be multiple output
    # files for a given input path.
    for path, path_df in df.groupby("path"):
        assert isinstance(path_df, pd.DataFrame), f"expected DataFrame, got {type(path_df)}"
        # create a hash to avoid output file collisions
        output_hash = hashlib.sha256(str(time.time()).encode()).hexdigest()[:16]
        path_str = path_to_str(path)
        outfile = os.path.join(output_dir, f"{path_str}-{output_hash}.parquet")
        logging.warning(f"writing file with {len(path_df)} rows to {outfile}")
        path_df.to_parquet(outfile)
        outfiles.append(outfile)
    return {"files": outfiles}


def get_parallelism(num_cores, available_memory, partition_size):
    max_partitions_by_cores = num_cores
    max_partitions_by_memory = available_memory // partition_size
    return min(max_partitions_by_cores, max_partitions_by_memory)


def main(data_dir: str,
         max_shards: Optional[int] = None,
         output_dir: str = "./ray-tmp/dedup/content_hash_index",
         force_parallelism: Optional[int] = None,
         read_mem_per_worker_gb: float = 16,
         ):
    data_dir = os.path.abspath(data_dir)
    output_dir = os.path.abspath(output_dir)

    os.makedirs(output_dir, exist_ok=True)
    fileglob = os.path.join(os.path.abspath(data_dir), "**", "*.parquet")
    print(f"listing files matching {fileglob}")
    files = glob.glob(fileglob, recursive=True)
    print(f"got {len(files)} files matching {fileglob}")
    if not len(files):
        return

    if max_shards is not None and len(files) > max_shards:
        print(f"[INFO] downsampling files to {max_shards} shards (got {len(files)})")
        files = files[:max_shards]

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

    start = time.time()

    ctx = ray.data.DataContext.get_current()
    ctx.execution_options.verbose_progress = True
    ctx.max_errored_blocks = 1000

    files_enum: Dict[str, int] = {f: i for i, f in enumerate(files)}

    def _add_file_number(row: Dict[str, Any]) -> Dict[str, Any]:
        row["pathnum"] = files_enum[row["path"]]
        return row

    # Note: only reading in the key makes this script way way way faster.
    # Setting 'include_paths=True' creates a column called 'path' containing the filename
    # from which each row originated.
    ds = ray.data.read_parquet(files,
                               columns=["key", ],
                               include_paths=True,
                               ray_remote_args={"num_cpus": 1, "memory": read_mem_per_worker_gb * BYTES_PER_GB}
                               )
    print(f"[INFO] finished reading files")
    print(f"[INFO] schema is {ds.schema()}")
    print(f"[INFO] ds is {ds}")

    # Sorting + mapping over batches of size 10k twice achieves deduplication of batch_size**2 elements.
    # Since we know that the most-frequent tables in TabLib occur 10**7 times, a batch size of 10**4
    # achieves deduplication of up to 10**8 total duplicates, ensuring complete deduplication at the
    # level of content_hash.
    ds = (ds.repartition(parallelism)
          .map(add_content_hash)
          .map(_add_file_number)
          .sort(["sortkey1", "sortkey2"]).map_batches(drop_duplicates, batch_size=10_000)
          .sort(["sortkey1", "sortkey2"]).map_batches(drop_duplicates, batch_size=10_000)
          .sort("pathnum"))

    output = (ds.select_columns(['key', 'path', 'content_hash'])
              .map_batches(write_file_output, fn_kwargs={"output_dir": output_dir})
              .take_all())

    print(f"Finished! {len(output)} results are written to {output_dir}")

    end = time.time()
    print(f"execution finished in {int(end - start)}s.")


if __name__ == "__main__":
    fire.Fire(main)
