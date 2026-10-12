# PI-SDFL

Official implementation and reproducibility package for:

**PI-SDFL: Trust-Gated Physics-Informed Sequential Decentralized Federated Learning for Reliable PEM Electrolyzer Prediction**

## Overview

PI-SDFL is a serverless sequential federated learning framework that uses
engineering knowledge during both local training and model acceptance.

The framework combines:

- Hard non-negativity output enforcement
- Current-hydrogen monotonicity regularization
- Anchor regularization
- Physics-sensitive trust gating with rollback

Experiments cover:

- A controlled PEM electrolyzer benchmark
- Measured NLR megawatt-scale PEM electrolyzer data
- Measured wind-farm data for cross-domain validation

## Repository Structure

```text
PI-SDFL/
├── src_elect/          Controlled PEM experiments
├── src_nlr/            NLR PEM experiments
├── src_wind/           Wind-farm experiments
├── data/               Local dataset directory
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

Generated datasets, experiment outputs, figures, cache files, and external
raw datasets are not tracked in the repository.

## Requirements

Python 3.11 was used for the reported experiments.

Install the required packages with:

```bash
pip install -r requirements.txt
```

## Dataset Setup

### Controlled PEM Benchmark

The controlled PEM benchmark is generated programmatically from
first-principles electrochemical and thermodynamic relationships based on
Faraday's law.

No external dataset download is required.

From the repository root, run:

```bash
python src_elect/generate_controlled_pem.py
```

The generated files are stored automatically in:

```text
data/pem_raw/
```

The generated datasets are:

```text
client1_data.csv
client2_data.csv
client3_data.csv
client4_data.csv
global_val.csv
global_test.csv
pooled_data.csv
master_pool.csv
```

### NLR PEM Dataset

Measured PEM electrolyzer data are obtained from the National Laboratory of
the Rockies Data Catalog.

The study uses the following four public datasets:

1. **NLR Historical Wind**
   - Dataset: Public Reference Data for Megawatt-Scale Hydrogen Electrolysis - NLR Historical Wind
   - DOI: 10.7799/3024096
   - File used: `combined_historical_wind_experiments.csv`

2. **NLR Historical Solar PV**
   - Dataset: Public Reference Data for Megawatt-Scale Hydrogen Electrolysis - NLR Historical Solar PV
   - DOI: 10.7799/3372313
   - File used: `combined_solarPV_experiments.csv`

3. **Simulated Wind**
   - Dataset: Public Reference Data for Megawatt-Scale Hydrogen Electrolysis - Simulated Wind
   - DOI: 10.7799/3007835
   - File used: `combined_wind_experiments.csv`

4. **Simulated Marine Hydrokinetic Tidal Turbine**
   - Dataset: Public Reference Data for Megawatt-Scale Hydrogen Electrolysis - Simulated Marine Hydrokinetic Tidal Turbine
   - DOI: 10.7799/3364185
   - File used: `combined_tidal_experiments.csv`

Place the four downloaded files in:

```text
data/nlr_raw/
```

Then run:

```bash
python src_nlr/run_nlr_preprocessing.py
```

Processed files are generated locally and are not included in this repository.

### Wind-Farm Dataset

Cross-domain validation uses:

**Inland-Offshore Wind Farm Dataset1**  
Yu Ding, 2021  
DOI: 10.5281/zenodo.5516552

Download:

```text
Inland_Offshore_Wind_Farm_Dataset1.zip
```

The archive contains data associated with turbines WT1-WT6. The PI-SDFL
experiments use WT1, WT2, WT3, and WT4.

Extract the required turbine files and place them in:

```text
data/wind_raw/
```

Then run:

```bash
python src_wind/run_wind_preprocessing.py
```

Raw NLR and wind datasets are not redistributed in this repository. They
should be obtained directly from their official public repositories.

## Running Experiments

### Controlled PEM

Generate the benchmark first:

```bash
python src_elect/generate_controlled_pem.py
```

Then run:

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

After downloading and preprocessing the NLR datasets:

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

After preprocessing the wind dataset:

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

The final experiments use 10 independent random seeds and 20 communication
rounds.

Dataset splitting, preprocessing, physics probes, model configurations, and
trust-gate settings are implemented in the corresponding experiment
directories.

Experiment outputs are generated automatically when the code is executed.

## Citation

If you use this repository, please cite the associated PI-SDFL paper.

Full citation information will be added upon publication.

## License

The PI-SDFL source code is released under the MIT License.

The NLR and wind datasets remain subject to the licenses and terms of their
respective data providers.
