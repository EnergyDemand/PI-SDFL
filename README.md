# PI-SDFL: Trust-Gated Physics-Informed Sequential Decentralized Federated Learning

**PI-SDFL: Trust-Gated Physics-Informed Sequential Decentralized Federated Learning for Reliable Hydrogen Production Prediction in Distributed PEM Electrolyzer Systems**

Submitted to *Advanced Engineering Informatics* (AEI), Elsevier.

---

## Overview

PI-SDFL is a serverless sequential federated learning framework for distributed PEM electrolyzer systems. It combines four mechanisms to prevent physics-knowledge degradation during sequential ring traversal:

- Hard non-negativity guard (current-factored Softplus output)
- Monotonicity regularization (L_mono)
- Anchor regularization
- Physics-sensitive trust gate with rollback

The framework is evaluated on three complementary settings:
- **Synthetic PEM**: controlled severity sweep and ablation study
- **NLR real PEM**: real 1.25-MW electrolyzer data, clean validation and sensor-gain fault robustness
- **Wind farm**: cross-domain generalization under a below-rated wind-speed–power monotonicity prior

---

## Repository Structure

```
pi_sdfl_aei/
├── src_elect/          Synthetic PEM experiment runners
├── src_nlr/            NLR PEM experiment runners
├── src_wind/           Wind farm experiment runners
├── scripts/            Figure generation and CSV export scripts
├── results/            Exported CSV result summaries (generated)
├── Figures/            Generated PDF figures (generated)
├── data/               Raw and processed datasets (not tracked — see below)
├── requirements.txt
└── README.md
```

---

## Requirements

```
torch>=2.0
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
matplotlib>=3.7
```

Install:
```bash
pip install -r requirements.txt
```

---

## Datasets

### Synthetic PEM
Generated from first-principles PEM physics (Faraday's law). Data is created by the preprocessing step embedded in `experiment_utils.py`. Place processed files in `data/`.

### NLR Real PEM
Download from the NLR Public Data Catalog:
- `combined_historical_wind_experiments.csv`
- `combined_solarPV_experiments.csv`
- `combined_wind_experiments.csv`
- `combined_simulated_tidal_experiments.csv`

Place in `data/nlr_raw/`, then run:
```bash
python src_nlr/run_nlr_preprocessing.py
```

### Wind Farm
Download the Inland Wind Farm Dataset (WT1–WT4 CSV files).  
Place in `data_wind_aei/raw/`, then run:
```bash
python src_wind/run_wind_preprocessing.py
```

---

## Running Experiments

Each directory contains a standalone runner per method. The execution order for each dataset is:

```
smoke → pilot → final
```

### Synthetic PEM

```bash
cd src_elect

# Clean comparison (7 methods)
python run_local_only.py --mode final
python run_centralized.py --mode final
python run_vanilla_sequential.py --mode final
python run_cwc.py --mode final
python run_pi_vanilla_sequential.py --mode final
python run_fedavg.py --mode final
python run_pi_sdfl.py --mode final
python run_frozen_reference.py --mode final
python aggregate_clean_results.py

# Severity sweep
python run_severity_sweep.py --mode final

# Ablation
python run_ablation.py --mode final
python aggregate_clean_results.py
```

### NLR PEM

```bash
cd src_nlr

# Clean
python run_nlr_local_only.py --mode final
python run_nlr_centralized.py --mode final
python run_nlr_vanilla_sequential.py --mode final
python run_nlr_cwc.py --mode final
python run_nlr_pi_vanilla_sequential.py --mode final
python run_nlr_fedavg.py --mode final
python run_nlr_pi_sdfl.py --mode final
python aggregate_nlr_results.py

# Sensor-gain fault
python run_nlr_sensor_fault.py --mode final --gains 1.1 1.2
```

### Wind Farm

```bash
cd src_wind

python run_wind_local_only.py --mode final
python run_wind_centralized.py --mode final
python run_wind_vanilla_sequential.py --mode final
python run_wind_cwc.py --mode final
python run_wind_pi_vanilla_sequential.py --mode final
python run_wind_fedavg.py --mode final
python run_wind_pi_sdfl.py --mode final
python run_wind_frozen_reference.py --mode final
python aggregate_wind_results.py
```

---

## Exporting Results to CSV

After all experiments complete:

```bash
python scripts/export_results_csv.py
```

This writes six CSV files to `results/`:

| File | Contents |
|---|---|
| `pem_clean_comparison.csv` | Synthetic PEM: 8 methods, R², RMSE, PVR, PC-R², gate |
| `pem_severity_sweep.csv` | Synthetic PEM: 5 frac levels × 4 methods |
| `pem_ablation.csv` | Synthetic PEM: 6 ablation variants |
| `nlr_clean_comparison.csv` | NLR PEM: 7 methods, R², RMSE (kg/h), PVR |
| `nlr_sensor_fault.csv` | NLR PEM: 4 methods × 2 gain levels |
| `wind_clean_comparison.csv` | Wind: 8 methods, R², RMSE (% rated), PVR |

---

## Generating Figures

```bash
python scripts/fig1_clean_comparison.py
python scripts/fig2_severity_sweep.py
python scripts/fig3_nlr_results.py
python scripts/fig4_ablation.py          # run after ablation completes
python scripts/fig5_wind_comparison.py
```

Figures are saved to `Figures/` as PDF files.

---

## Key Implementation Notes

**Evaluation protocol:**  
Smoke and pilot modes evaluate on audit partitions. Final mode evaluates on held-out test partitions. This is enforced via `_EVAL_MODE` in each `experiment_utils` file.

**Centralized evaluation head:**  
After pooled training, the Centralized baseline evaluates using a fresh head fitted on pooled calibration data — the same protocol as all other shared-body methods.

**PI-Vanilla-Sequential:**  
The wind implementation includes the corrected version where L_mono is active throughout ring training (the ESWA submission had a bug that silently disabled it).

**Physics probe:**  
Synthetic PEM uses a Cartesian 50×5×5 grid. NLR and wind use an empirical multi-context probe (5 contexts × 50 physics-feature values = 250 points).

**TAU_PVR:**  
Set to 0.05 for PEM (both synthetic and NLR). Set to 0.10 for wind to account for higher natural variability in SCADA measurements.

---

## Citation

If you use this code, please cite the associated paper (citation details to be added upon acceptance).

---

## License

MIT License. See `LICENSE` for details.
