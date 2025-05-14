!pip install sdv pandas numpy matplotlib # Ensure sdv is installed

import pandas as pd
import numpy as np
from sdv.sequential import PARSynthesizer
import matplotlib.pyplot as plt
import time # To measure training time

# --- Configuration ---
# Assume 'df' is your ORIGINAL combined DataFrame of ~150 days
# Ensure it has a DatetimeIndex
# df = pd.read_csv(...) # Load your data if not already in memory
# df['date'] = pd.to_datetime(df['date'])
# df = df.set_index('date').sort_index()

# Assume these lists are defined from your previous script:
# target_cols = [...]
# past_covariate_cols = [...]
# future_covariate_cols = [...]

# How many days of synthetic data to generate?
# Let's aim for roughly the same amount as the original data
n_synthetic_days = len(df) # e.g., 150

# SDV Training parameters (adjust as needed, more epochs usually better but slower)
sdv_epochs = 150 # Start with fewer epochs for speed, increase for quality (e.g., 300-500)
sdv_batch_size = 32 # Adjust based on memory/performance

# --- 1. Prepare Data for SDV ---
# Select only target and past covariates for synthesizer training
# Reset index so 'date' becomes a regular column needed by SDV
cols_for_synthesizer = target_cols + past_covariate_cols
df_for_sdv = df[cols_for_synthesizer].reset_index()

print(f"Original data shape for SDV training: {df_for_sdv.shape}")
print(f"Columns for SDV training: {cols_for_synthesizer}")

# Ensure no NaNs before fitting (fill conservatively)
df_for_sdv_filled = df_for_sdv.ffill().bfill()
if df_for_sdv_filled.isnull().sum().sum() > 0:
    print("Warning: NaNs remain after ffill/bfill. Consider imputation.")
    # Handle remaining NaNs more robustly if necessary, e.g., df_for_sdv_filled.fillna(0)

# --- 2. Define Metadata for SDV ---
# Define types for the columns being synthesized
metadata_fields = {
    "date": {"type": "datetime", "format": "%Y-%m-%d"}, # Adjust format if needed
    # Define target columns (assuming numeric)
    **{col: {"type": "numerical", "subtype": "float"} for col in target_cols},
    # Define past covariates (assuming numeric)
    **{col: {"type": "numerical", "subtype": "float"} for col in past_covariate_cols},
}

metadata = {
    "fields": metadata_fields,
    "sequence_index": "date",
    # No sequence_key needed if it's one continuous series
}

# --- 3. Initialize and Train Synthesizer ---
synthesizer = PARSynthesizer(metadata=metadata,
                           epochs=sdv_epochs,
                           batch_size=sdv_batch_size,
                           verbose=True)

print("\nFitting SDV Synthesizer on original data...")
start_time = time.time()
synthesizer.fit(df_for_sdv_filled)
end_time = time.time()
print(f"Synthesizer training complete. Time taken: {end_time - start_time:.2f} seconds")

# --- 4. Generate Synthetic Target and Past Covariates ---
print(f"\nGenerating {n_synthetic_days} steps of synthetic targets and past covariates...")
# We want one continuous sequence following the original data
# Note: PAR may sometimes generate slightly different lengths; check output.
synthetic_tp_sequences = synthesizer.sample(num_sequences=1,
                                            sequence_length=n_synthetic_days)

# Select the first sequence if multiple were generated unexpectedly
if isinstance(synthetic_tp_sequences, list):
     synthetic_tp_df = synthetic_tp_sequences[0]
else:
     synthetic_tp_df = synthetic_tp_sequences

# Check length
if len(synthetic_tp_df) != n_synthetic_days:
    print(f"Warning: Generated sequence length ({len(synthetic_tp_df)}) differs from requested ({n_synthetic_days}). Using generated length.")
    n_synthetic_days = len(synthetic_tp_df) # Update count based on actual output

print(f"Generated {len(synthetic_tp_df)} synthetic steps for targets/past covs.")

# --- 5. Generate Synthetic Dates and Future Covariates ---
last_real_date = df.index[-1]
synthetic_dates = pd.date_range(start=last_real_date + pd.Timedelta(days=1),
                                periods=n_synthetic_days, freq='D')

# Create a DataFrame for synthetic future covariates
synthetic_future_df = pd.DataFrame(index=synthetic_dates)

print("Generating synthetic future covariates based on date logic...")
if future_covariate_cols:
    for col in future_covariate_cols:
        # !! IMPORTANT: REPLACE THIS WITH YOUR ACTUAL LOGIC for generating each future covariate !!
        if 'day_of_week' in col or 'weekday' in col:
            synthetic_future_df[col] = synthetic_dates.dayofweek
        elif 'day_of_year' in col:
             synthetic_future_df[col] = synthetic_dates.dayofyear
        elif 'month' in col:
            synthetic_future_df[col] = synthetic_dates.month
        elif 'year' in col:
            synthetic_future_df[col] = synthetic_dates.year
        elif 'week_of_year' in col:
             synthetic_future_df[col] = synthetic_dates.isocalendar().week.astype(int)
        elif 'is_holiday' in col:
            # You need a holiday calendar relevant to your data
            # Example: Check if date is in a predefined holiday list/calendar
            # requires external library like 'holidays'
            # Example using a dummy list:
            # my_holidays = ['2024-01-01', '2024-07-04'] # Example
            # synthetic_future_df[col] = synthetic_dates.strftime('%Y-%m-%d').isin(my_holidays).astype(int)
             print(f"Placeholder logic for '{col}'. Needs real implementation.")
             synthetic_future_df[col] = 0 # Placeholder - REPLACE
        elif col.startswith('future_'): # Catch-all for other future vars
             print(f"Placeholder logic for '{col}'. Needs real implementation.")
             synthetic_future_df[col] = 0 # Placeholder - REPLACE
        else:
            print(f"Warning: Unhandled future covariate '{col}'. Assigning 0.")
            synthetic_future_df[col] = 0
else:
    print("No future covariates defined.")

# --- 6. Combine Synthetic Parts ---
# Set the date index on the synthetic target/past covariate data
synthetic_tp_df = synthetic_tp_df.set_index('date')
synthetic_tp_df.index = synthetic_dates # Assign the correct dates

# Combine synthetic targets/past with synthetic future covariates
synthetic_df_complete = pd.concat([synthetic_tp_df, synthetic_future_df], axis=1)

# Ensure columns are in the same order as the original DataFrame
synthetic_df_complete = synthetic_df_complete[df.columns]

print("\nCombined Synthetic Data Sample:")
print(synthetic_df_complete.head())

# --- 7. Create Augmented DataFrame ---
augmented_df = pd.concat([df, synthetic_df_complete], axis=0).sort_index()

print(f"\nOriginal data shape:   {df.shape}")
print(f"Synthetic data shape:  {synthetic_df_complete.shape}")
print(f"Augmented data shape:  {augmented_df.shape}")

# --- 8. Verification and Evaluation (Crucial!) ---
print("\nVerifying augmented data...")
print(f"NaN check in augmented_df: {augmented_df.isnull().sum().sum()}")
if augmented_df.isnull().sum().sum() > 0:
    print("NaNs found in augmented data. Consider imputation or review generation.")
    print(augmented_df.isnull().sum()[augmented_df.isnull().sum() > 0])
    # Simple fill as fallback:
    # augmented_df = augmented_df.ffill().bfill()

print(f"Augmented date range: {augmented_df.index.min()} to {augmented_df.index.max()}")
print(f"Total days in augmented dataset: {len(augmented_df)}")

# --- Visual Comparison (Example for one target variable) ---
target_to_plot = target_cols[0]
plt.figure(figsize=(15, 7))
plt.plot(df.index, df[target_to_plot], label='Original Data', color='blue')
plt.plot(synthetic_df_complete.index, synthetic_df_complete[target_to_plot], label='Synthetic Data', color='red', alpha=0.8)
plt.title(f'Original vs. Synthetic Time Series ({target_to_plot})')
plt.ylabel(target_to_plot)
plt.xlabel("Date")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# Compare distributions
plt.figure(figsize=(10, 5))
df[target_to_plot].hist(alpha=0.6, label='Original', bins=30, density=True)
synthetic_df_complete[target_to_plot].hist(alpha=0.6, label='Synthetic', bins=30, density=True)
plt.title(f'Distribution Comparison ({target_to_plot})')
plt.legend()
plt.show()

# --- Ready for Training ---
print("\n'augmented_df' is ready.")
# Now you can use 'augmented_df' in your Darts TFT training pipeline
# Replace the 'df' variable in your original training script with 'augmented_df'
# Example: df = augmented_df # In your TFT script
# Then proceed with splitting, scaling (FIT scaler ONLY on train split of augmented data!), etc.