# COVID-19 Dynamics & Vaccination Modeling in Israel

Data science and mathematical modeling project by **Andrey Makushev** and **Maria Jubran**.

## Overview

This project analyzes COVID-19 dynamics in Israel using real-world data from **Our World in Data**. The analysis compares the pre-vaccination and post-vaccination periods, combines statistical analysis with compartmental epidemic models, and explores several counterfactual scenarios.

The project includes:

- daily-to-weekly data aggregation
- descriptive analysis of cases, deaths, vaccination, and government stringency
- correlation analysis
- multiple linear regression
- SIRD modeling for 2020 waves
- SIVRD modeling for 2021 waves
- nonlinear least-squares parameter estimation with multiple starting points
- scenario simulations for transmission and vaccination conditions

The full written analysis is included in [`report.pdf`](report.pdf).

## Data

The analysis uses the **Our World in Data COVID-19 dataset**, filtered to Israel.

The script expects the dataset file to be named:

```text
owid-covid-data.csv
```

Place that file in the repository root before running the analysis.

The main variables used are:

- new and cumulative cases
- new and cumulative deaths
- government stringency index
- people vaccinated
- people fully vaccinated
- booster totals
- population

Daily observations are aggregated into weekly time series.

## Study Periods

The project separates the data into two main periods:

- **2020:** pre-vaccination period
- **2021:** vaccination period

Each period is additionally divided into two epidemic waves, producing four wave-specific model fits.

## Statistical Analysis

The project compares weekly COVID-19 indicators across 2020 and 2021 and uses multiple linear regression to model weekly deaths.

Reported results in the project include:

- **2020 regression:** R² = **0.591**
- **2021 regression:** R² = **0.897**
- In the 2021 model, the fully vaccinated rate had a negative coefficient with weekly deaths after accounting for the other included variables.

These results are interpreted in the accompanying report within the scope of the project's observational analysis.

## SIRD and SIVRD Models

Two compartmental models are used:

### SIRD

Used for the 2020 waves:

- **S** — Susceptible
- **I** — Infected
- **R** — Recovered
- **D** — Deceased

### SIVRD

Used for the 2021 waves and extends SIRD with:

- **V** — Vaccinated

The fitted parameters are:

- `beta` — transmission rate
- `gamma` — recovery rate
- `mu` — mortality rate
- `nu` — vaccination rate in the SIVRD model

Parameters are estimated with `scipy.optimize.least_squares` using the trust-region reflective method, a soft-L1 loss, weighted residuals, and multiple random starting points.

## Scenario Analysis

The project evaluates several model-based scenarios:

- **Base**
- **High transmission**
- **Low transmission**
- **Optimal response**
- **No vaccination** for the 2021 SIVRD waves

The scenarios compare changes in modeled peak infections and total deaths under different parameter assumptions.

## Repository Structure

```text
covid19-dynamics-modeling/
├── README.md
├── covid19_dynamics_modeling.py
├── report.pdf
├── requirements.txt
└── .gitignore
```

## Requirements

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

## Running the Project

1. Download the Our World in Data COVID-19 dataset.
2. Save it in the repository root as `owid-covid-data.csv`.
3. Install the dependencies.
4. Run:

```bash
python covid19_dynamics_modeling.py
```

The script generates descriptive statistics, regression output, model fits, fitted parameters, and scenario plots.

## Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- SciPy
- Statsmodels
- tqdm

## Authors

**Andrey Makushev**  
**Maria Jubran**