import logging
import os
import uuid
from dataclasses import dataclass
from typing import Dict, Any

import pyarrow as pa
import ray

from tabliblib.io import read_arrow_bytes


@ray.remote
def write_dataframe_to_file(row: Dict[str, Any],
                            root_dir: str,
                            output_format: str,
                            ) -> Dict[str, Any]:
    """
    A Ray remote function that writes a DataFrame to a CSV file.

    Parameters:
    - data: The dictionary containing the DataFrame.
    - base_path: The base directory to write the CSV files.
    - index: The index of the data item, used to generate a unique filename.
    """
    # Ensure the base directory exists
    os.makedirs(root_dir, exist_ok=True)

    # Generate a unique filename for each DataFrame
    df_uuid = str(uuid.uuid1())
    arrow_bytes = row["arrow_bytes"]
    if arrow_bytes is not None:
        df = read_arrow_bytes(row["arrow_bytes"], raise_on_error=True)
    else:
        return row

    output_file = "__".join((str(row["content_hash"]), df_uuid)) + "." + output_format
    filename = os.path.join(os.path.abspath(root_dir), output_file)

    logging.warning(f"[DEBUG] writing dataframe of shape {df.shape} to {filename}")
    # Write DataFrame to CSV
    if output_format == "csv":
        df.to_csv(filename, index=False)
    elif output_format == "parquet":
        # TODO(jpgard): this still fails sometimes due to very large strings; we probably
        #  need to also check the total length of a row in filter_rows_too_long (as opposed
        #  to only the length of each individual cell).
        df.to_parquet(filename, index=False)
    return row


@dataclass
class DataFrameFileDataSink:
    base_path: str
    output_format: str
    mem_per_writer: int
    num_cpus_per_writer: int = 1

    def write(self, dataset):
        """
        Writes each element of the dataset to a separate CSV file.

        Parameters:
        - dataset: The Ray Dataset to process.
        """
        # Use map_batches to apply the write function to each dataset element in parallel
        return dataset.map(self._write_element)

    def _write_element(self, element):
        """
        Helper function to write a row of data to file.
        """
        try:
            # Dispatch Ray tasks to write each element in the batch to a CSV file
            future = (write_dataframe_to_file
                      .options(num_cpus=self.num_cpus_per_writer,
                               memory=self.mem_per_writer)
                      .remote(element, self.base_path, self.output_format))

            # Wait for all tasks to complete and return their filenames
            return ray.get(future)
        except pa.lib.ArrowNotImplementedError as e:
            # Handles 'pyarrow.lib.ArrowNotImplementedError:
            # Cannot write struct type 'meta' with no child field to Parquet. Consider adding a dummy child field.'
            logging.warning(f"pa.lib.ArrowNotImplementedError raised writing element with "
                            f"content_hash {element['content_hash']}; {e}")
            return element

        except Exception as e:
            logging.warning(f"exception raised writing element with content_hash {element['content_hash']}: {e}")
            return element
