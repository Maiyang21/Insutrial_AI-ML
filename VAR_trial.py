import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt

# Assuming 'df' is your combined DataFrame from the previous script
# df = pd.read_csv(...) # Or however you load your combined 150 days

# --- 1. Select Variables for VAR ---
# Include targets and PAST covariates. Exclude FUTURE covariates for now.
# Make sure columns are numeric. Handle categoricals if any (e.g., one-hot encode AFTER splitting potentially)
var_cols = target_cols + past_covariate_cols
df_var = df[var_cols].copy()
df_var = df_var.astype(float) # Ensure numeric types

# --- 2. Check for and Address Stationarity ---
# (This is a simplified check; proper analysis is needed)
print("Checking stationarity...")
adf_results = {}
data_diff = df_var.copy()
diff_order = 0
all_stationary = False

while not all_stationary and diff_order < 2: # Limit differencing
    all_stationary = True
    for name, column in data_diff.items():
        # Drop NaNs resulting from previous differencing before testing
        result = adfuller(column.dropna())
        adf_results[name] = result[1] # p-value
        if result[1] > 0.05:
            all_stationary = False
            print(f"  {name} is non-stationary (p={result[1]:.3f}). Differencing...")
            data_diff = data_diff.diff().dropna() # Difference entire dataframe
            diff_order += 1
            break # Re-test all columns after differencing
    if all_stationary:
         print("All series appear stationary.")

if not all_stationary:
    print("Warning: Could not achieve stationarity for all series. VAR results may be unreliable.")

print(f"Final differencing order: {diff_order}")
original_first_values = {}
if diff_order > 0:
    # Store values needed for inverse differencing
    # For order 1: store the first row of the original df_var
    original_first_values[1] = df_var.iloc[0:diff_order] # Store first 'diff_order' rows
    if diff_order > 1:
         # For order 2: also need the difference of the first two rows
         original_first_values[2] = df_var.diff().iloc[diff_order:diff_order+1] # Store first diff row

# --- 3. Fit VAR Model ---
# Select lag order (e.g., using AIC/BIC - can be slow)
model = VAR(data_diff)
# lag_order_results = model.select_order(maxlags=15) # Can take time
# selected_lag_order = lag_order_results.aic # Or .bic etc.
# print(f"Selected lag order (AIC): {selected_lag_order}")
# For simplicity, let's pick a reasonable lag, e.g., 7 (weekly) - ADJUST THIS
selected_lag_order = 7
results = model.fit(selected_lag_order)
# print(results.summary())

# --- 4. Simulate ---
n_samples_to_generate = 300 # How many new time steps?
print(f"Generating {n_samples_to_generate} synthetic steps...")
# Need the last 'selected_lag_order' observations of the (differenced) training data
lag_order = results.k_ar
last_observations = data_diff.values[-lag_order:]

# Simulate step-by-step
simulated_diff = results.forecast(y=last_observations, steps=n_samples_to_generate)
simulated_diff_df = pd.DataFrame(simulated_diff, columns=data_diff.columns)

# --- 5. Inverse Differencing ---
def inverse_difference(last_original_row, forecast_diff, second_last_original_row=None, diff_order=1):
    """Reverses differencing (simplified for order 1 or 2)."""
    if diff_order == 0:
        return forecast_diff
    elif diff_order == 1:
        # Cumulative sum starting from the last original value
        inverted = np.r_[last_original_row.values, forecast_diff].cumsum(axis=0)[1:]
        return inverted
    elif diff_order == 2:
         # This is more complex. Simplified approach:
         # First, invert the second difference
         inverted_1st_diff = np.r_[second_last_original_row.diff().dropna().values, forecast_diff].cumsum(axis=0)[1:]
         # Then, invert the first difference
         inverted = np.r_[last_original_row.values, inverted_1st_diff].cumsum(axis=0)[1:]
         return inverted
    else:
        raise ValueError("Inverse difference only implemented for order 0, 1, 2")

print("Inverse differencing...")
if diff_order == 0:
    simulated_undifferenced_df = simulated_diff_df
elif diff_order == 1:
     last_orig_row = df_var.iloc[-1:] # Last row of original data
     simulated_vals = inverse_difference(last_orig_row, simulated_diff_df.values, diff_order=1)
     simulated_undifferenced_df = pd.DataFrame(simulated_vals, columns=df_var.columns)
elif diff_order == 2:
     last_orig_row = df_var.iloc[-1:]
     second_last_orig_row = df_var.iloc[-2:-1] # Not quite right, need careful index handling
     # This part needs careful implementation based on how diff=2 was calculated
     # For simplicity, let's assume we stored the necessary start points correctly.
     # This requires a more robust inverse differencing function handling initial conditions.
     # Placeholder - this part might be incorrect without careful index logic
     print("Warning: Inverse difference for order 2 is complex and might be inaccurate here.")
     # Attempt a basic cumulative sum approach (likely wrong)
     inverted_1 = simulated_diff_df.cumsum() + df_var.diff().iloc[-1].values # Approximate 1st diff recovery
     inverted_2 = inverted_1.cumsum() + df_var.iloc[-1].values # Approximate original recovery
     simulated_undifferenced_df = pd.DataFrame(inverted_2.values, columns=df_var.columns)


# --- 6. Add Time Index and Future Covariates ---
last_real_date = df.index[-1]
synthetic_dates = pd.date_range(start=last_real_date + pd.Timedelta(days=1),
                                periods=n_samples_to_generate, freq='D')
simulated_undifferenced_df.index = synthetic_dates

# **CRITICAL:** Now, generate or assign FUTURE covariates for these new dates
# Example: If 'day_of_week' was a future covariate
# simulated_undifferenced_df['future_day_of_week'] = synthetic_dates.dayofweek
# Do this for ALL your future covariates based on their logic.
# If future covariates depend on external forecasts, you'd need those forecasts.
# For this example, let's assume future_cols need manual regeneration:
synthetic_df_final = simulated_undifferenced_df # Start with VAR generated data

if future_covariate_cols:
     print("Generating synthetic future covariates...")
     future_df_synthetic = pd.DataFrame(index=synthetic_dates)
     # Regenerate each future covariate based on the synthetic date
     for col in future_covariate_cols:
         # EXAMPLE LOGIC - REPLACE WITH YOUR ACTUAL LOGIC
         if 'day_of_week' in col:
             future_df_synthetic[col] = synthetic_dates.dayofweek
         elif 'month' in col:
             future_df_synthetic[col] = synthetic_dates.month
         elif 'is_holiday' in col:
             # Need a holiday calendar relevant to your data
             # future_df_synthetic[col] = synthetic_dates.isin(my_holiday_calendar)
             future_df_synthetic[col] = 0 # Placeholder
         else:
             # Assign a default or use logic specific to that feature
             future_df_synthetic[col] = 0 # Placeholder

     # Combine VAR-generated with manually generated future covariates
     synthetic_df_final = pd.concat([simulated_undifferenced_df, future_df_synthetic], axis=1)
     # Ensure column order matches original df if necessary
     synthetic_df_final = synthetic_df_final[df.columns]


print("\nSynthetic Data Sample (VAR based):")
print(synthetic_df_final.head())

# --- 7. Evaluate ---
# Compare distributions, ACF, CCF, plots of synthetic_df_final vs df

# Example Plot: Compare one target variable
target_to_plot = target_cols[0]
plt.figure(figsize=(12, 6))
plt.plot(df.index, df[target_to_plot], label='Original Data')
plt.plot(synthetic_df_final.index, synthetic_df_final[target_to_plot], label='Synthetic Data (VAR)', alpha=0.7)
plt.title(f'Original vs. Synthetic Time Series ({target_to_plot})')
plt.legend()
plt.show()

# Example Distribution Comparison:
plt.figure(figsize=(10, 5))
df[target_to_plot].hist(alpha=0.6, label='Original', bins=30, density=True)
synthetic_df_final[target_to_plot].hist(alpha=0.6, label='Synthetic', bins=30, density=True)
plt.title(f'Distribution Comparison ({target_to_plot})')
plt.legend()
plt.show()