# PI-SDFL

Official implementation and reproducibility package for:

**PI-SDFL: Trust-Gated Physics-Informed Sequential Decentralized Federated Learning for Reliable PEM Electrolyzer Prediction**

## Overview

PI-SDFL is a serverless sequential federated learning framework that uses engineering knowledge during both local training and model acceptance.

The framework combines:

- Hard non-negativity output enforcement
- Current–hydrogen monotonicity regularization
- Anchor regularization
- Physics-sensitive trust gating with rollback

Experiments cover:

- Controlled PEM electrolyzer benchmark
- Measured NLR megawatt-scale PEM electrolyzer data
- Measured wind-farm data for cross-domain validation

## Repository Structure

```text
PI-SDFL/
├── src_elect/          Controlled PEM experiments
├── src_nlr/            NLR PEM experiments
├── src_wind/           Wind-farm experiments
├── requirements.txt
├── run_pi_sdfl.py
├── .gitignore
└── README.md
```

Generated results, figures, cache files, and external raw datasets are not tracked in the repository.

## Requirements

Python 3.11 was used for the reported experiments.

Install the required packages with:

```bash
pip install -r requirements.txt
```

## Dataset Setup

### Controlled PEM Benchmark

The controlled PEM benchmark is generated programmatically from first-principles electrochemical relationships based on Faraday's law.

No external dataset download is required. The generation and experiment code is provided in:

```text
src_elect/
```

### NLR PEM Dataset

The measured megawatt-scale PEM electrolyzer data are publicly available through the National Laboratory of the Rockies Data Catalog.

Place the downloaded data files in:

```text
data/nlr_raw/
```

Then run:

```bash
python src_nlr/run_nlr_preprocessing.py
```

### Wind-Farm Dataset

The Inland Wind Farm Dataset is publicly available through Zenodo.

Place the required WT1–WT4 data files in:

```text
data_wind_aei/raw/
```

Then run:

```bash
python src_wind/run_wind_preprocessing.py
```

Raw external datasets are not redistributed in this repository.

## Running Experiments

### Controlled PEM

```bash
python src_elect/run_local_only.py --mode final
python src_elect/run_centralized.py --mode final
python src_elect/run_fedavg.py --mode final
python src_elect/run_vanilla_sequential.py --mode final
python src_elect/run_cwc.py --mode final
python src_elect/run_pi_vanilla_sequential.py --mode final
python src_elect/run_frozen_reference.py --mode final
python src_elect/run_pi_sdfl.py --mode final
python src_elect/run_severity_sweep.py --mode final
python src_elect/run_ablation.py --mode final
```

### NLR PEM

```bash
python src_nlr/run_nlr_local_only.py --mode final
python src_nlr/run_nlr_centralized.py --mode final
python src_nlr/run_nlr_fedavg.py --mode final
python src_nlr/run_nlr_vanilla_sequential.py --mode final
python src_nlr/run_nlr_cwc.py --mode final
python src_nlr/run_nlr_pi_vanilla_sequential.py --mode final
python src_nlr/run_nlr_pi_sdfl.py --mode final
python src_nlr/run_nlr_sensor_fault.py --mode final --gains 1.1 1.2
```

### Wind Farm

```bash
python src_wind/run_wind_local_only.py --mode final
python src_wind/run_wind_centralized.py --mode final
python src_wind/run_wind_fedavg.py --mode final
python src_wind/run_wind_vanilla_sequential.py --mode final
python src_wind/run_wind_cwc.py --mode final
python src_wind/run_wind_pi_vanilla_sequential.py --mode final
python src_wind/run_wind_frozen_reference.py --mode final
python src_wind/run_wind_pi_sdfl.py --mode final
```

## Reproducibility

The final experiments use 10 independent random seeds and 20 communication rounds.

Dataset splitting, preprocessing, physics probes, model configurations, and trust-gate settings are implemented in the corresponding experiment directories.

Experiment outputs are generated automatically when the code is executed.

## Citation

If you use this repository, please cite the associated PI-SDFL paper.

Full citation information will be added upon publication.
