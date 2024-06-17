# tabliblib

`tabliblib` is a Python library for processing data from [TabLib](https://arxiv.org/abs/2310.07875).
`tabliblib` supports table-, row-, and column-level filtering of tables to extract
a high-quality pool of tabular data from the raw TabLib corpus.

`tabliblib` was used to build [The Tremendous TabLib Trawl (T4)](https://huggingface.co/datasets/mlfoundations/t4-full), 
the dataset used to train [TabuLa-8B](https://huggingface.co/mlfoundations/tabula-8b), 
which is the state-of-the-art zero- and few-shot tabular prediction foundation model 
described in our paper "Large-Scale Transfer Learning for Tabular Data via Language Modeling".

<div align=center>
<img alt="few-shot results curve" src="https://github.com/mlfoundations/tabliblib/blob/main/assets/all_tasks_curves.png" width=50%>
</div>

*This is an alpha release of `tabliblib`!* We expect the API to evolve over time
and do not guarantee that it will remain stable.
We are providing this alpha version in an effort to make our dataset creation process
transparent and reproducible for the community.

The T4 dataset can be accessed on Hugging Face [here](https://huggingface.co/datasets/mlfoundations/t4-full).

The authors of `tabliblib` are not affiliated with the creation of TabLib or Approximate Labs
(although we are grateful to the creators of TabLib for their contributions to the community!).

# Quickstart

First, set up your environment using either the provided Conda `environment.yml`
file or `requirements.txt`.

You will also need access to TabLib. You can request access 
[here](https://huggingface.co/datasets/approximatelabs/tablib-v1-full). 

The `scripts` directory contains scripts that can be used to process TabLib.
An end-to-end tablib processing pipeline works in a two stages:
1. *Deduplication* (script: `deduplicate_tablib.py`): We build an index that lists which tables from which tablib 
shards should be included in the final datasets. The script `deduplicate_tablib.py` 
builds this index and writes it to a series of parquet files. This ensures that the
second stage of the process avoids redundant processing of duplicate tables,
and that each table appears only once in the output.
```shell
# start a local ray cluster
ray start --head
# run the deduplication script
python scripts/dedup_tablib.py \
    --data_dir "/path/to/tablib/" \
    --output_dir "/path/to/output/content_hash_index/myjob/" \
    --force-parallelism 1000
```
2. *Filtering and processing* (script: `process_tablib.py`): We read the deduplicated data, apply a series of filtering rules, 
and write the resulting tables to individual parquet files.
In addition to filtering at the **table** level, this script also performs filtering at the 
**column** and **row** levels. You can implement a custom config in `tabliblib.config.py`
to control the filtering process, or use `v6` to replicat the filtering used to construct T4.
```shell
# start a local ray cluster if it is not already running
ray start --head
# run the deduplication script; set --dedup_dir to the output of your deduplication pipeline from above
python scripts/process_tablib.py \
    --data_dir "/path/to/tablib/" \
    --dedup_dir "/path/to/output/content_hash_index/myjob/" \
    --output_dir /path/to/output/ \
    --config_version v6 \
    --chunk_size 512 \
    --chunk_index 0
```
Note that this script should be performed once per chunk, from `0..NUM_CHUNKS`,
where `NUM_CHUNKS` is `ceil(76213/chunk_size)` (there are 76213 input shards in TabLib).


# Notes on Ray Data performance

`tabliblib` makes use of [Ray Data](https://docs.ray.io/en/latest/data/overview.html) to execute
efficient distributed data processing pipelines. This means that `tabliblib` runs effectively 
on machines with many (i.e., dozens) of CPUs.

For best performance whenn running `process_tablib.py`, we recommend not processing all of 
TabLib in a single Ray pipeline. Instead, we suggest running multiple concurrent pipelines, 
each with a different value of the `--chunk_index` flag. This splits the files in TabLib
into separate nonoverlapping chunks and processes them separately, avoiding Ray performance
degradation that we observed when the input data exceeds a certain critical threshold
(this threshold will vary depending on hardware characteristics)
and helps to ensure smooth scaling and execution. 
This chunking also allows for only partial processing of TabLib 
(because the TabLib data is shuffled before chunking), which avoids processing
all of TabLib when only a smaller sample is needed.

# Language detection model

If using language detection capabilities in `process_tablib.py`,
you need to manually download the fasttext language detection model from [here](
https://fasttext.cc/docs/en/language-identification.html) and place it at `./tabliblib/`

You can do this via:

```shell
wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin && mv lid.176.bin tabliblib
```

# Training models using the results of a `tabliblib` pipeline

`tabliblib` produces a set of Parquet files, which can be read, manipulated, and processed
with any downstream library capable of natively reading Parquet.

For training [TabuLa-8B](https://huggingface.co/mlfoundations/tabula-8b), 
we used the [`rtfm`](https://github.com/mlfoundations/rtfm) library, which contains utilities
to serialize Parquet files into text, construct supervised prediction tasks
from the tables, and train a language model for tabular tasks.

# Additional Resources

Some additional resources relevant to `tabliblib`:

* Our paper, "Large Scale Transfer Learning for Tabular Data via Language Modeling"
* The [t4 dataset](https://huggingface.co/datasets/mlfoundations/t4-full) on Hugging Face (used to train TabuLa-8B)
* The TabuLa-8B [evaluation suite data](https://huggingface.co/datasets/mlfoundations/tabula-8b-eval-suite) on Hugging
  Face
* The [TabuLa-8B model](https://huggingface.co/mlfoundations/tabula-8b) on Hugging Face
* [`rtfm`](https://github.com/mlfoundations/rtfm), a toolkit for training tabular foundation models