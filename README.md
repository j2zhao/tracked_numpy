# Summary of Experiments

This repository contains experiments for the DSLog system. It includes a modified version of NumPy that has the data type 'tracked_float', allowing for provenance tracking at a low-level for NumPy Arrays. It also requires the download of the TurboRangeEncoder (https://github.com/powturbo/Turbo-Range-Coder) code, and DuckDB/Parquet libraries.

## Important Files

- compression.py : Contains main compression algorithm
- compression_algs.py : Contains code for evaluating compression experiments
- numpy_lineage.py/example_query_image.py/example_query_relational.py : Contains code for generating example query pipelines
- query_experiments.py : Contains code for evaluating query experiments
- prov_functions.py : Contains code for evaluating reuse experiments
