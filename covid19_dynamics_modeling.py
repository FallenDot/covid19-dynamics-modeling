"""
COVID-19 Dynamics and Vaccination Modeling in Israel

Authors: Andrey Makushev and Maria Jubran

This script reproduces the analysis used in the accompanying project report:
- Our World in Data preprocessing and weekly aggregation
- descriptive statistics and correlation analysis
- multiple linear regression
- SIRD / SIVRD compartmental modeling
- multi-start nonlinear least-squares parameter fitting
- scenario / what-if simulations

The analytical methodology and model structure are preserved from the original project.
The script expects `owid-covid-data.csv` in the working directory.
"""

# =======================================================
# 1) IMPORTS AND DEPENDENCIES
# =======================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import statsmodels.api as sm
from scipy.integrate import odeint
from scipy.optimize import least_squares


# =======================================================
# 2) DATA LOADING AND CLEANING
# =======================================================

def load_and_clean_data(csv_file='owid-covid-data.csv'):
    """
    Loads the OWID COVID-19 dataset and filters it for Israel.
    Cleans and prepares relevant columns, handles missing values, etc.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame filtered up to 2023-11-04
    """
    # Set random seed for reproducible multi-start optimization
    np.random.seed(42)
    
    # Load dataset
    data = pd.read_csv(csv_file)

    # Filter only for Israel
    israel_data = data[data['location'] == 'Israel'].copy()

    # Keep only relevant columns
    required_columns = [
        'date', 'new_cases', 'new_deaths', 'total_cases', 'total_deaths',
        'stringency_index', 'people_vaccinated', 'people_fully_vaccinated',
        'total_boosters', 'population'
    ]
    israel_data = israel_data[required_columns]

    # Convert date column to datetime
    israel_data['date'] = pd.to_datetime(israel_data['date'])

    # Filter dataset up to 2023-11-04
    cutoff_date = pd.Timestamp('2023-11-04')
    israel_data = israel_data[israel_data['date'] <= cutoff_date]

    # Fill missing stringency_index as 0 after 2022-12-31
    israel_data['stringency_index'] = israel_data['stringency_index'].fillna(0)

    # Handle vaccination data
    start_vaccine_date = pd.Timestamp('2020-12-19')
    cum_cols = ['people_vaccinated', 'people_fully_vaccinated', 'total_boosters']
    before_vaccine_mask = israel_data['date'] < start_vaccine_date
    israel_data.loc[before_vaccine_mask, cum_cols] = israel_data.loc[before_vaccine_mask, cum_cols].fillna(0)
    israel_data[cum_cols] = israel_data[cum_cols].fillna(method='ffill')

    return israel_data


def convert_to_weekly(israel_data):
    """
    Converts cleaned daily data to weekly aggregates.
    
    Returns:
        pd.DataFrame: Weekly aggregated DataFrame
    """
    # Create weekly periods ending on Sunday
    israel_data['week'] = israel_data['date'].dt.to_period('W-SUN').dt.to_timestamp()

    # Aggregate data to weekly level
    weekly_data = israel_data.groupby('week').agg({
        'new_cases': 'sum',
        'new_deaths': 'sum',
        'total_cases': 'last',
        'total_deaths': 'last',
        'stringency_index': 'mean',
        'people_vaccinated': 'last',
        'people_fully_vaccinated': 'last',
        'total_boosters': 'last',
        'population': 'last'
    }).reset_index()

    # Calculate rates
    weekly_data['vaccination_rate'] = (weekly_data['people_vaccinated'] / weekly_data['population']) * 100
    weekly_data['fully_vaccinated_rate'] = (weekly_data['people_fully_vaccinated'] / weekly_data['population']) * 100

    return weekly_data


def filter_periods(weekly_data):
    """
    Splits the weekly data into the study periods and epidemic waves used in the analysis.
    
    Returns:
        dict: A dictionary of period_name -> DataFrame
    """
    # Define time periods
    period_1_start = pd.Timestamp('2020-02-03')
    period_1_end   = pd.Timestamp('2020-12-07')
    period_2_start = pd.Timestamp('2020-12-14')
    period_2_end   = pd.Timestamp('2021-12-27')
    
    # Wave boundaries used in the project
    period_1_1_end   = pd.Timestamp('2020-05-11')
    period_1_2_start = pd.Timestamp('2020-05-18')
    period_2_1_end   = pd.Timestamp('2021-05-24')
    period_2_2_start = pd.Timestamp('2021-05-31')

    # Filter
    period_1_data = weekly_data[(weekly_data['week'] >= period_1_start) & (weekly_data['week'] <= period_1_end)]
    period_2_data = weekly_data[(weekly_data['week'] >= period_2_start) & (weekly_data['week'] <= period_2_end)]
    period_wave_1_data = weekly_data[(weekly_data['week'] >= period_1_start) & (weekly_data['week'] <= period_1_1_end)]
    period_wave_2_data = weekly_data[(weekly_data['week'] >= period_1_2_start) & (weekly_data['week'] <= period_1_end)]
    period_wave_3_data = weekly_data[(weekly_data['week'] >= period_2_start) & (weekly_data['week'] <= period_2_1_end)]
    period_wave_4_data = weekly_data[(weekly_data['week'] >= period_2_2_start) & (weekly_data['week'] <= period_2_end)]

    # Return as dict for convenience
    return {
        'period_1_data': period_1_data,
        'period_2_data': period_2_data,
        'wave_1': period_wave_1_data,
        'wave_2': period_wave_2_data,
        'wave_3': period_wave_3_data,
        'wave_4': period_wave_4_data
    }


# =======================================================
# 3) PLOTTING & DESCRIPTIVE FUNCTIONS
# =======================================================

def plot_cases_and_deaths(df, title_suffix="", cases_label="New Cases", deaths_label="New Deaths (x100)"):
    """
    Plots weekly new cases and new deaths in the same figure.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(df['week'], df['new_cases'], label=cases_label, color='blue')
    plt.plot(df['week'], df['new_deaths'] * 100, label=deaths_label, color='red')
    plt.xlabel("Week")
    plt.ylabel("Count")
    plt.title(f"Weekly New Cases & Deaths {title_suffix}")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_stringency_index(df, title_suffix=""):
    """
    Plots stringency index over time.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(df['week'], df['stringency_index'], label="Stringency Index", color='purple')
    plt.xlabel("Week")
    plt.ylabel("Index (0-100)")
    plt.title(f"Government Stringency Index Over Time {title_suffix}")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_vaccination_progress(df, title_suffix=""):
    """
    Plots vaccination progress over time.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(df['week'], df['vaccination_rate'], label="At least 1 dose", color='green')
    plt.plot(df['week'], df['fully_vaccinated_rate'], label="Fully vaccinated", color='orange')
    plt.xlabel("Week")
    plt.ylabel("Percentage of Population (%)")
    plt.title(f"COVID-19 Vaccination Progress {title_suffix}")
    plt.legend()
    plt.grid(True)
    plt.show()


def descriptive_stats(df, columns, period_name=""):
    """
    Prints descriptive statistics for selected columns.
    """
    print(f"===== {period_name} Descriptive Statistics =====")
    print(df[columns].describe())


def compare_averages(period_1_df, period_2_df):
    """
    Compare average weekly cases, deaths, and stringency index between two DataFrames.
    """
    comparison_df = pd.DataFrame({
        "Metric": [
            "Avg. Weekly Cases", 
            "Avg. Weekly Deaths", 
            "Avg. Stringency Index", 
            "Avg. Vaccination Rate", 
            "Avg. Fully Vaccinated Rate"
        ],
        "2020": [
            period_1_df['new_cases'].mean(),
            period_1_df['new_deaths'].mean(),
            period_1_df['stringency_index'].mean(),
            0,  # No vaccinations in 2020
            0
        ],
        "2021": [
            period_2_df['new_cases'].mean(),
            period_2_df['new_deaths'].mean(),
            period_2_df['stringency_index'].mean(),
            period_2_df['vaccination_rate'].mean(),
            period_2_df['fully_vaccinated_rate'].mean()
        ]
    })
    print("===== Comparison of Averages Between 2020 & 2021 =====")
    print(comparison_df)

    # Plot bar comparisons
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    sns.barplot(x=["2020", "2021"], 
                y=[period_1_df['new_cases'].mean(), period_2_df['new_cases'].mean()],
                ax=axes[0])
    axes[0].set_title("Comparison of Avg. Weekly Cases")

    sns.barplot(x=["2020", "2021"], 
                y=[period_1_df['new_deaths'].mean(), period_2_df['new_deaths'].mean()],
                ax=axes[1])
    axes[1].set_title("Comparison of Avg. Weekly Deaths")

    sns.barplot(x=["2020", "2021"], 
                y=[period_1_df['stringency_index'].mean(), period_2_df['stringency_index'].mean()],
                ax=axes[2])
    axes[2].set_title("Comparison of Avg. Stringency Index")

    plt.tight_layout()
    plt.show()


# =======================================================
# 4) CORRELATION AND MULTIPLE LINEAR REGRESSION (MLR)
# =======================================================

def plot_correlation_matrix(df, cols, title="Correlation Matrix"):
    """
    Plots a correlation matrix for specified columns.
    """
    corr = df[cols].corr()
    plt.figure(figsize=(6, 5))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title(title)
    plt.show()


def run_mlr(df, y_var, x_vars, label=""):
    """
    Runs multiple linear regression using statsmodels OLS and prints the model summary.
    """
    X = df[x_vars]
    X = sm.add_constant(X)
    y = df[y_var]

    model = sm.OLS(y, X).fit()
    print(f"\n===== MLR Results: {label} =====")
    print(model.summary())
    return model


# =======================================================
# 5) SIRD / SIVRD MODELING
# =======================================================

def sird_model(state, t, beta, gamma, mu):
    """SIRD model differential equations: [S, I, R, D]."""
    S, I, R, D = state
    N = S + I + R + D
    
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - (gamma + mu) * I
    dRdt = gamma * I
    dDdt = mu * I
    return [dSdt, dIdt, dRdt, dDdt]


def sivrd_model(state, t, beta, gamma, mu, nu):
    """SIVRD model differential equations: [S, I, V, R, D]."""
    S, I, V, R, D = state
    N = S + I + V + R + D
    
    dSdt = -beta * S * I / N - nu * S
    dIdt = beta * S * I / N - (gamma + mu) * I
    dVdt = nu * S
    dRdt = gamma * I
    dDdt = mu * I
    return [dSdt, dIdt, dVdt, dRdt, dDdt]


def sird_residuals(params, data, t, scales=None):
    """Residuals for SIRD model fitting."""
    beta, gamma, mu = params

    I0 = data['new_cases'].iloc[1]
    D0 = 0
    R0 = 0
    N = data['population'].iloc[0]
    S0 = N - I0 - R0 - D0
    
    solution = odeint(sird_model, [S0, I0, R0, D0], t, args=(beta, gamma, mu))

    if scales is None:
        scales = {
            'cases': np.std(data['new_cases']) or 1.0,
            'deaths': np.std(data['total_deaths']) or 1.0
        }

    cases_resid = (solution[:, 1] - data['new_cases']) / scales['cases']
    deaths_resid = (solution[:, 3] - data['total_deaths']) / scales['deaths']

    return np.concatenate([cases_resid, deaths_resid])


def sivrd_residuals(params, data, t, scales=None):
    """Residuals for SIVRD model fitting."""
    beta, gamma, mu, nu = params

    I0 = data['new_cases'].iloc[1]
    V0 = data['people_vaccinated'].iloc[0]
    D0 = data['total_deaths'].iloc[0]
    R0 = 0
    N = data['population'].iloc[0]
    S0 = N - I0 - V0 - R0 - D0
    
    solution = odeint(sivrd_model, [S0, I0, V0, R0, D0], t, args=(beta, gamma, mu, nu))

    if scales is None:
        scales = {
            'cases': np.std(data['new_cases']) or 1.0,
            'deaths': np.std(data['total_deaths']) or 1.0,
            'vaccinated': np.std(data['people_vaccinated']) or 1.0
        }
    cases_resid = (solution[:, 1] - data['new_cases']) / scales['cases']
    deaths_resid = (solution[:, 4] - data['total_deaths']) / scales['deaths']
    vacc_resid = (solution[:, 2] - data['people_vaccinated']) / scales['vaccinated']

    return np.concatenate([cases_resid, deaths_resid, vacc_resid])


def multi_start_fit(residual_func, data, n_starts=20, model_type="SIRD"):
    """
    Performs the project's multi-start, unbounded least-squares optimization for SIRD or SIVRD.
    Returns the best result from scipy.optimize.least_squares.
    """
    t = np.arange(len(data))
    param_count = 3 if model_type == "SIRD" else 4

    # Precompute scales once
    scales = {
        'cases': np.std(data['new_cases']) or 1.0,
        'deaths': np.std(data['total_deaths']) or 1.0
    }
    if model_type == "SIVRD":
        scales['vaccinated'] = np.std(data['people_vaccinated']) or 1.0

    best_result = None
    best_cost = np.inf

    for _ in tqdm(range(n_starts), desc=f"Fitting {model_type} model"):
        x0 = np.random.uniform(low=0.0, high=1.0, size=param_count)
        try:
            result = least_squares(
                residual_func, 
                x0, 
                args=(data, t, scales), 
                method='trf', 
                loss='soft_l1'
            )
            if result.cost < best_cost:
                best_cost = result.cost
                best_result = result
        except Exception as e:
            print(f"Warning: Optimization failed for x0={x0}: {str(e)}")
            continue

    if best_result is None:
        raise ValueError("All optimization attempts failed.")

    return best_result


def fit_sird(data, n_starts=20):
    """Convenience function to fit SIRD model."""
    return multi_start_fit(sird_residuals, data, n_starts, model_type="SIRD")


def fit_sivrd(data, n_starts=20):
    """Convenience function to fit SIVRD model."""
    return multi_start_fit(sivrd_residuals, data, n_starts, model_type="SIVRD")


def get_model_predictions(data, params, model_type="SIRD"):
    """
    Runs the fitted ODE system and returns the full solution array.
    """
    t = np.arange(len(data))

    if model_type == "SIRD":
        beta, gamma, mu = params
        I0 = data['new_cases'].iloc[1]
        D0 = 0
        R0 = 0
        N  = data['population'].iloc[0]
        S0 = N - I0 - R0 - D0

        solution = odeint(sird_model, [S0, I0, R0, D0], t, args=(beta, gamma, mu))

    else:  # SIVRD
        beta, gamma, mu, nu = params
        I0 = data['new_cases'].iloc[1]
        V0 = data['people_vaccinated'].iloc[0]
        D0 = data['total_deaths'].iloc[0]
        R0 = 0
        N  = data['population'].iloc[0]
        S0 = N - I0 - V0 - R0 - D0

        solution = odeint(sivrd_model, [S0, I0, V0, R0, D0], t, args=(beta, gamma, mu, nu))
    
    return solution


def plot_model_fit(data, solution, model_type="SIRD"):
    """
    Plots SIRD or SIVRD model predictions against the observed data.
    """
    plt.figure(figsize=(15, 10))

    # Subplot 1: Cases
    plt.subplot(3, 1, 1)
    plt.plot(data.index, data['new_cases'], 'b.', alpha=0.5, label='Actual Cases')
    plt.plot(data.index, solution[:, 1], 'b-', label='Predicted Cases')
    plt.legend()
    plt.title(f"{model_type} Model Fit - Cases")
    plt.grid(True)

    # Subplot 2: Deaths
    plt.subplot(3, 1, 2)
    death_idx = 3 if model_type == "SIRD" else 4
    plt.plot(data.index, data['total_deaths'], 'r.', alpha=0.5, label='Actual Deaths')
    plt.plot(data.index, solution[:, death_idx], 'r-', label='Predicted Deaths')
    plt.legend()
    plt.title(f"{model_type} Model Fit - Deaths")
    plt.grid(True)

    # Subplot 3: Vaccinations (SIVRD only)
    if model_type == "SIVRD":
        plt.subplot(3, 1, 3)
        plt.plot(data.index, data['people_vaccinated'], 'g.', alpha=0.5, label='Actual Vaccinated')
        plt.plot(data.index, solution[:, 2], 'g-', label='Predicted Vaccinated')
        plt.legend()
        plt.title("Vaccination Progress")
        plt.grid(True)

    plt.tight_layout()
    plt.show()


def run_models(period_1_data, period_2_data, n_starts=20):
    """
    Fits the project's unbounded SIRD and SIVRD models, plots the fits,
    and returns the optimization results.
    """
    print("Fitting SIRD model to 'period_1_data' (unbounded)...")
    sird_result = fit_sird(period_1_data, n_starts)
    sird_solution = get_model_predictions(period_1_data, sird_result.x, model_type="SIRD")
    plot_model_fit(period_1_data, sird_solution, "SIRD")

    print("\nFitting SIVRD model to 'period_2_data' (unbounded)...")
    sivrd_result = fit_sivrd(period_2_data, n_starts)
    sivrd_solution = get_model_predictions(period_2_data, sivrd_result.x, model_type="SIVRD")
    plot_model_fit(period_2_data, sivrd_solution, "SIVRD")

    print("\n===== Final Fitted Parameters =====")
    print("SIRD:", sird_result.x, "Cost:", sird_result.cost)
    print("SIVRD:", sivrd_result.x, "Cost:", sivrd_result.cost)

    return sird_result, sivrd_result


# =======================================================
# 6) SCENARIO / WHAT-IF ANALYSIS
# =======================================================

def run_what_if_scenarios(data, params, model_type="SIRD", scenario="base"):
    """
    Applies the project's parameter changes for a scenario and runs the model forward.
    """
    t = np.arange(len(data))
    
    if model_type == "SIRD":
        beta, gamma, mu = params

        if scenario == "high_transmission":
            beta *= 1.3
        elif scenario == "low_transmission":
            beta *= 0.7
        elif scenario == "optimal":
            beta = min(beta, 2.0)  # or some logic
            gamma = max(gamma, 0.3)
            mu = min(mu, 0.005)

        # Initial states
        I0 = data['new_cases'].iloc[1]
        D0 = 0
        R0 = 0
        N  = data['population'].iloc[0]
        S0 = N - I0 - R0 - D0

        solution = odeint(sird_model, [S0, I0, R0, D0], t, args=(beta, gamma, mu))
        params_used = (beta, gamma, mu)

    else:  # SIVRD
        beta, gamma, mu, nu = params

        if scenario == "no_vaccination":
            nu = 0
        elif scenario == "high_transmission":
            beta *= 1.3
        elif scenario == "low_transmission":
            beta *= 0.7
        elif scenario == "optimal":
            beta = min(beta, 2.0)
            gamma = max(gamma, 0.3)
            mu = min(mu, 0.005)
            nu = max(nu, 0.05)

        I0 = data['new_cases'].iloc[1]
        V0 = data['people_vaccinated'].iloc[0]
        D0 = data['total_deaths'].iloc[0]
        R0 = 0
        N  = data['population'].iloc[0]
        S0 = N - I0 - V0 - R0 - D0

        solution = odeint(sivrd_model, [S0, I0, V0, R0, D0], t, args=(beta, gamma, mu, nu))
        params_used = (beta, gamma, mu, nu)

    return solution, params_used


def plot_scenario_comparison(data, base_solution, scenario_solution, model_type="SIRD", scenario_name="", wave_num=1):
    """
    Plots the base-model trajectory against a selected scenario.
    """
    weeks = np.arange(len(data))
    plt.figure(figsize=(15, 10))
    plt.suptitle(f"{model_type} Model - Wave {wave_num} - {scenario_name} Scenario", fontsize=14, y=1.02)

    # Cases
    plt.subplot(2, 1, 1)
    plt.plot(weeks, data['new_cases'], 'k.', alpha=0.5, label='Actual Cases')
    plt.plot(weeks, base_solution[:, 1], 'b-', label='Base Model Cases')
    plt.plot(weeks, scenario_solution[:, 1], 'r--', label=f"{scenario_name} Cases")
    plt.xlabel('Week')
    plt.ylabel('Cases')
    plt.title(f"{scenario_name} Scenario - Cases")
    plt.legend()
    plt.grid(True)

    # Deaths
    plt.subplot(2, 1, 2)
    death_idx = 3 if model_type == "SIRD" else 4
    plt.plot(weeks, data['total_deaths'], 'k.', alpha=0.5, label='Actual Deaths')
    plt.plot(weeks, base_solution[:, death_idx], 'b-', label='Base Model Deaths')
    plt.plot(weeks, scenario_solution[:, death_idx], 'r--', label=f"{scenario_name} Deaths")
    plt.xlabel('Week')
    plt.ylabel('Deaths')
    plt.title(f"{scenario_name} Scenario - Deaths")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def analyze_scenarios(data_list, params_list, model_types):
    """
    Runs the project scenarios for each wave, prints summary statistics, and plots comparisons.
    """
    scenarios = ["no_vaccination", "high_transmission", "low_transmission", "optimal"]

    for wave_idx, (df, param, mtype) in enumerate(zip(data_list, params_list, model_types), start=1):
        # Get base solution
        base_solution = get_model_predictions(df, param, mtype)

        print(f"\n{'='*40}\nAnalyzing Wave {wave_idx} ({mtype})\n{'='*40}")
        
        for sc in scenarios:
            # Skip 'no_vaccination' for SIRD
            if sc == "no_vaccination" and mtype == "SIRD":
                continue

            scenario_sol, scenario_params = run_what_if_scenarios(df, param, mtype, scenario=sc)
            
            # Show basic stats
            base_peak_cases = np.max(base_solution[:, 1])
            scenario_peak_cases = np.max(scenario_sol[:, 1])
            death_idx = 3 if mtype == "SIRD" else 4
            base_deaths = base_solution[-1, death_idx]
            scenario_deaths = scenario_sol[-1, death_idx]

            print(f"\nScenario: {sc}")
            print(f"  Base Peak Cases:     {base_peak_cases:,.0f}")
            print(f"  Scenario Peak Cases: {scenario_peak_cases:,.0f}")
            print(f"  Base Total Deaths:   {base_deaths:,.0f}")
            print(f"  Scenario Deaths:     {scenario_deaths:,.0f}")

            plot_scenario_comparison(df, base_solution, scenario_sol,
                                     model_type=mtype, scenario_name=sc, wave_num=wave_idx)


# =======================================================
# 7) MAIN EXECUTION
# =======================================================

if __name__ == "__main__":
    # Load and clean the OWID dataset
    raw_data = load_and_clean_data('owid-covid-data.csv')
    weekly_data = convert_to_weekly(raw_data)
    periods = filter_periods(weekly_data)

    # Extract analysis periods and waves
    period_1_data = periods['period_1_data']
    period_2_data = periods['period_2_data']
    wave_1_data   = periods['wave_1']
    wave_2_data   = periods['wave_2']
    wave_3_data   = periods['wave_3']
    wave_4_data   = periods['wave_4']

    # Descriptive stats
    descriptive_stats(period_1_data, ['new_cases', 'new_deaths', 'stringency_index'], "2020")
    plot_cases_and_deaths(period_1_data, title_suffix="(2020)")
    plot_stringency_index(period_1_data, title_suffix="(2020)")

    descriptive_stats(period_2_data, 
                      ['new_cases', 'new_deaths', 'vaccination_rate', 'fully_vaccinated_rate', 'stringency_index'],
                      "2021")
    plot_cases_and_deaths(period_2_data, title_suffix="(2021)")
    plot_vaccination_progress(period_2_data, title_suffix="(2021)")
    plot_stringency_index(period_2_data, title_suffix="(2021)")

    # Compare averages 2020 vs 2021
    compare_averages(period_1_data, period_2_data)

    # Correlation and multiple linear regression
    plot_correlation_matrix(period_1_data, ['new_cases','new_deaths','stringency_index'], "2020 Correlation")
    run_mlr(period_1_data, 'new_deaths', ['new_cases'], label="2020 MLR (No Vaccines)")

    plot_correlation_matrix(period_2_data, ['new_cases','new_deaths','stringency_index','vaccination_rate','fully_vaccinated_rate'],
                            "2021 Correlation")
    run_mlr(period_2_data, 'new_deaths', ['new_cases','stringency_index','vaccination_rate','fully_vaccinated_rate'],
            label="2021 MLR (With Vaccines)")

    # Fit SIRD and SIVRD models
    # Fit wave_1 (SIRD) with wave_3 (SIVRD), then wave_2 (SIRD) with wave_4 (SIVRD)
    sird_res_1, sivrd_res_1 = run_models(wave_1_data, wave_3_data, n_starts=20)
    sird_res_2, sivrd_res_2 = run_models(wave_2_data, wave_4_data, n_starts=20)

    # Run what-if scenarios
    # Collect fitted parameters from the optimization results
    wave_1_params = sird_res_1.x
    wave_2_params = sird_res_2.x
    wave_3_params = sivrd_res_1.x
    wave_4_params = sivrd_res_2.x

    data_list = [wave_1_data, wave_2_data, wave_3_data, wave_4_data]
    params_list = [wave_1_params, wave_2_params, wave_3_params, wave_4_params]
    model_types = ["SIRD", "SIRD", "SIVRD", "SIVRD"]

    analyze_scenarios(data_list, params_list, model_types)
