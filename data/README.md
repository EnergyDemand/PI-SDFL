# Dataset Setup

This directory is used for local dataset preparation.

Raw third-party datasets are not redistributed in this repository.

## Controlled PEM Benchmark

The controlled PEM benchmark is generated from the repository root using:

```bash
python src_elect/generate_controlled_pem.py
```

The generated files are saved in:

```text
data/pem_raw/
```

## NLR PEM Dataset

The measured PEM electrolyzer datasets used in this study are publicly available through the National Laboratory of the Rockies Data Catalog.

Place the downloaded files in:

```text
data/nlr_raw/
```

Then run:

```bash
python src_nlr/run_nlr_preprocessing.py
```

## Wind Farm Dataset

The wind-farm data used for cross-domain validation are publicly available through the Inland Wind Farm Dataset on Zenodo.

Place the required WT1-WT4 files in:

```text
data/wind_raw/
```

Then run:

```bash
python src_wind/run_wind_preprocessing.py
```

The controlled PEM data are generated locally. The NLR and wind datasets should be obtained from their original public repositories.
