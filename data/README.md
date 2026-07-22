# Datasets

This project does NOT ship the datasets (CICIDS2017 runs ~1.1GB and
UNSW-NB15 ~380MB as CSVs, and both have their own licenses). `data/raw/` is
git-ignored for that reason — GitHub hard-rejects any file over 100MB, so
these CSVs must never be committed. Download them yourself and put the CSVs
where config.yaml points; only this README is tracked in git.

## Quick start without downloading anything
Set `dataset: synthetic` in `config.yaml` (the default) and run the pipeline.
The synthetic generator produces class-imbalanced tabular data shaped like
network flows, so you can verify the whole system runs before fetching the
real data. It is not a substitute for real results.

## CICIDS2017 (primary)
Source: Canadian Institute for Cybersecurity (UNB)
  https://www.unb.ca/cic/datasets/ids-2017.html

Recommended: use the *corrected / improved* release, which fixes known
labelling and CICFlowMeter errors:
  Engelen et al., "Troubleshooting an Intrusion Detection Dataset" (2021).

Place the per-day CSVs in:
  data/raw/cicids2017/*.csv

The loader strips column-name whitespace, replaces Infinity with NaN, drops
identifier/leaky columns (Flow ID, IPs, source port, timestamp after the
split), and maps the Label column to binary (BENIGN -> 0, anything else -> 1).

## UNSW-NB15 (secondary, cross-dataset validation)
Source: UNSW Canberra
  https://research.unsw.edu.au/projects/unsw-nb15-dataset

Place the CSVs (e.g. the training/testing set CSVs that carry named columns,
including `label` and `attack_cat`) in:
  data/raw/unsw_nb15/*.csv

## NSL-KDD (optional third dataset)
Not wired by default. To add it, write a loader in src/data_loaders.py that
returns a DataFrame with a binary `Label`, add a constraint map in
src/constraints.py, and register it in config.yaml.

## Why two datasets
CICIDS2017 (CICFlowMeter, ~78 features) and UNSW-NB15 (Argus/Bro, a different
feature set and attack mix) are built with different toolchains. Confirming
findings on both shows results are not tied to one feature-extraction
pipeline. Retrain on each — do not transfer models across datasets.
