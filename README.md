# PI-SDFL

Official implementation and reproducibility package for:

**PI-SDFL: Trust-Gated Physics-Informed Sequential Decentralized Federated Learning for Reliable PEM Electrolyzer Prediction**

## Overview

PI-SDFL is a serverless sequential federated learning framework that uses
engineering knowledge during both local training and model acceptance.

The framework combines:

- hard non-negativity output enforcement
- current–hydrogen monotonicity regularization
- anchor regularization
- physics-sensitive trust gating with rollback

Experiments cover:

- a controlled PEM electrolyzer benchmark
- measured NLR megawatt-scale PEM electrolyzer data
- measured wind-farm data for cross-domain validation

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

Experiment outputs and figures are generated automatically and are not stored
in the repository.
Requirements
Python 3.11 was used for the reported experiments.
Install the required packages with:
pip install -r requirements.txt

Datasets
Controlled PEM Benchmark
The controlled PEM benchmark is generated from first-principles
electrochemical relationships based on Faraday's law. The required generation
and experiment code is included in src_elect/.
NLR PEM Dataset
The measured PEM data are publicly available through the National Laboratory
of the Rockies Data Catalog.
After obtaining the required files, run:
python src_nlr/run_nlr_preprocessing.py

Wind-Farm Dataset
The Inland Wind Farm Dataset is publicly available through Zenodo.
After obtaining the WT1–WT4 data files, run:
python src_wind/run_wind_preprocessing.py

Raw external datasets are not redistributed in this repository.
Running Experiments
Controlled PEM
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

NLR PEM
python src_nlr/run_nlr_local_only.py --mode final
python src_nlr/run_nlr_centralized.py --mode final
python src_nlr/run_nlr_fedavg.py --mode final
python src_nlr/run_nlr_vanilla_sequential.py --mode final
python src_nlr/run_nlr_cwc.py --mode final
python src_nlr/run_nlr_pi_vanilla_sequential.py --mode final
python src_nlr/run_nlr_pi_sdfl.py --mode final
python src_nlr/run_nlr_sensor_fault.py --mode final --gains 1.1 1.2

Wind Farm
python src_wind/run_wind_local_only.py --mode final
python src_wind/run_wind_centralized.py --mode final
python src_wind/run_wind_fedavg.py --mode final
python src_wind/run_wind_vanilla_sequential.py --mode final
python src_wind/run_wind_cwc.py --mode final
python src_wind/run_wind_pi_vanilla_sequential.py --mode final
python src_wind/run_wind_frozen_reference.py --mode final
python src_wind/run_wind_pi_sdfl.py --mode final

Reproducibility
The final experiments use 10 independent random seeds and 20 communication
rounds. Dataset splits, preprocessing, physics probes, model configurations,
and trust-gate settings are implemented in the corresponding experiment
directories.
Citation
If you use this repository, please cite the associated PI-SDFL paper.
Full citation information will be added upon publication.
