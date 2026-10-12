"""
generate_controlled_pem.py - PI-SDFL controlled PEM electrolyzer benchmark generator.

The benchmark is generated from first-principles electrochemical and
thermodynamic relationships.

Generated variables:
    voltage
    current
    temperature
    pressure
    h2_flow_rate
    oxygen_flow_rate

Active model inputs used in PI-SDFL:
    current
    temperature
    pressure

Prediction target:
    h2_flow_rate (mL/min)

Voltage is generated from the polarization relationship but is excluded
from the final model because of its strong collinearity with current.

Oxygen flow is retained only as an auxiliary variable and is not used as
an input because it is stoichiometrically determined by hydrogen flow.

Non-IID heterogeneity is created through client-specific operating ranges,
quantity skew, and graded measurement noise.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = PROJECT_DIR / "data" / "pem_raw"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Physical constants
# ============================================================

F = 96485.33       # Faraday constant (C/mol)
N_CELLS = 20       # Number of cells in the PEM stack
R = 8.314          # Universal gas constant (J/mol/K)


# ============================================================
# PEM data generation
# ============================================================

def generate_site(
    n,
    I_lo,
    I_hi,
    T_lo,
    T_hi,
    P_lo,
    P_hi,
    noise,
    seed,
):
    """Generate one controlled PEM client dataset."""

    rng = np.random.default_rng(seed)

    # Operating conditions
    I = rng.uniform(I_lo, I_hi, n)      # Current (A)
    T = rng.uniform(T_lo, T_hi, n)      # Temperature (degC)
    P = rng.uniform(P_lo, P_hi, n)      # Pressure (bar)

    T_K = T + 273.15

    # --------------------------------------------------------
    # Faraday efficiency
    # --------------------------------------------------------
    eta = np.clip(
        0.91 + 0.0035 * (T - 60.0),
        0.80,
        0.995,
    )

    # --------------------------------------------------------
    # Faraday's law
    # H2 molar production rate (mol/s)
    # --------------------------------------------------------
    n_h2 = eta * N_CELLS * I / (2.0 * F)

    # --------------------------------------------------------
    # Ideal-gas volumetric conversion
    # Output: mL/min
    # --------------------------------------------------------
    V_h2 = (
        n_h2
        * R
        * T_K
        / (P * 1e5)
        * 1e6
        * 60.0
    )

    # --------------------------------------------------------
    # Temperature-dependent polarization relationship
    # --------------------------------------------------------
    V_rev = 1.23 - 0.0009 * (T - 25.0)

    r_ohm = 0.0028 * (
        1.0 - 0.004 * (T - 60.0)
    )

    v_cell = (
        V_rev
        + r_ohm * I
        + 0.06 * np.log(np.maximum(I, 1.0))
    )

    V_stack = N_CELLS * v_cell

    # --------------------------------------------------------
    # Flow measurement noise
    # --------------------------------------------------------
    h2 = V_h2 * (
        1.0 + rng.normal(0.0, noise, n)
    )

    # Prevent non-positive physical flow
    h2 = np.maximum(h2, 1e-3)

    # --------------------------------------------------------
    # H2:O2 stoichiometry
    # --------------------------------------------------------
    o2 = h2 / 2.0

    return pd.DataFrame(
        {
            "voltage": V_stack,
            "current": I,
            "temperature": T,
            "pressure": P,
            "h2_flow_rate": h2,
            "oxygen_flow_rate": o2,
        }
    )


# ============================================================
# Main generation procedure
# ============================================================

def main():

    print("=" * 65)
    print("PI-SDFL Controlled PEM Benchmark Generation")
    print("=" * 65)

    # --------------------------------------------------------
    # Four heterogeneous clients
    # --------------------------------------------------------

    c1 = generate_site(
        n=2000,
        I_lo=30,
        I_hi=150,
        T_lo=50,
        T_hi=75,
        P_lo=2,
        P_hi=6,
        noise=0.020,
        seed=101,
    )

    c2 = generate_site(
        n=900,
        I_lo=25,
        I_hi=130,
        T_lo=45,
        T_hi=72,
        P_lo=2,
        P_hi=7,
        noise=0.035,
        seed=102,
    )

    c3 = generate_site(
        n=600,
        I_lo=20,
        I_hi=140,
        T_lo=40,
        T_hi=78,
        P_lo=1,
        P_hi=6,
        noise=0.050,
        seed=103,
    )

    c4 = generate_site(
        n=500,
        I_lo=15,
        I_hi=120,
        T_lo=40,
        T_hi=70,
        P_lo=1,
        P_hi=5,
        noise=0.070,
        seed=104,
    )

    # Centralized reference pool
    pooled = pd.concat(
        [c1, c2, c3, c4],
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Independent global evaluation sets
    # --------------------------------------------------------

    global_val = generate_site(
        n=1000,
        I_lo=15,
        I_hi=150,
        T_lo=40,
        T_hi=78,
        P_lo=1,
        P_hi=7,
        noise=0.030,
        seed=201,
    )

    global_test = generate_site(
        n=2000,
        I_lo=15,
        I_hi=150,
        T_lo=40,
        T_hi=78,
        P_lo=1,
        P_hi=7,
        noise=0.030,
        seed=202,
    )

    # Optional reusable benchmark reservoir
    master_pool = generate_site(
        n=20000,
        I_lo=15,
        I_hi=150,
        T_lo=40,
        T_hi=78,
        P_lo=1,
        P_hi=7,
        noise=0.030,
        seed=203,
    )

    # --------------------------------------------------------
    # Save datasets
    # --------------------------------------------------------

    outputs = {
        "client1_data.csv": c1,
        "client2_data.csv": c2,
        "client3_data.csv": c3,
        "client4_data.csv": c4,
        "global_val.csv": global_val,
        "global_test.csv": global_test,
        "pooled_data.csv": pooled,
        "master_pool.csv": master_pool,
    }

    for filename, dataframe in outputs.items():

        path = OUTPUT_DIR / filename

        dataframe.to_csv(
            path,
            index=False,
        )

        print(
            f"Saved: {path} "
            f"({len(dataframe):,} samples)"
        )

    print()
    print("Generation complete.")
    print(f"Output directory: {OUTPUT_DIR}")
    print()
    print("Active PI-SDFL inputs: current, temperature, pressure")
    print("Target: h2_flow_rate")
    print("=" * 65)


if __name__ == "__main__":
    main()