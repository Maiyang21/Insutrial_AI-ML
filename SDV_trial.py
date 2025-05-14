import pandas as pd
from sdv.sequential import PARSynthesizer
import os
import matplotlib.pyplot as plt

# Assuming 'df' is your combined DataFrame with DatetimeIndex
# Get all CSV files in the current directory
csv_files = [file for file in os.listdir() if file.endswith('.csv')]

# Read and combine all CSV files into a single DataFrame
df_list = [pd.read_csv(file) for file in csv_files]
df = pd.concat(df_list, ignore_index=True)
df = pd.read_csv() # Or however you load your combined 150 days
# Reset index so 'date' becomes a column for SDV
df_sdv = df.reset_index()

# --- 1. Define Metadata ---
# Helps SDV understand the data structure
metadata = {
    "fields": {
        "date": {"type": "datetime", "format": "%Y-%m-%d"}, # Adjust format if needed
        # Define target columns (assuming numeric)
        **{col: {"type": "numerical", "subtype": "float"} for col in target_cols},
        # Define past covariates (assuming numeric)
        **{col: {"type": "numerical", "subtype": "float"} for col in past_covariate_cols},
        # Define future covariates (treat as context or numeric/categorical)
        # Option 1: Treat as regular numeric/categorical features
        **{col: {"type": "numerical", "subtype": "integer"} for col in future_covariate_cols if 'day' in col or 'month' in col} # Example
        # Add other future covariates based on their type (e.g., boolean for holidays)
    },
    "sequence_index": "date",
    # If you had multiple independent entities (e.g., stores, sensors), you'd use sequence_key
    # "sequence_key": "entity_id_column",
}

# --- 2. Initialize and Train Synthesizer ---
# PARSynthesizer is often a good choice for sequences
# You can experiment with PAR hyperparameters (epochs, batch_size, hidden_dim etc.)
synthesizer = PARSynthesizer(metadata=metadata,
                           epochs=100, # Reduce epochs for quick test, increase for quality (e.g., 300-500+)
                           verbose=True)

print("Fitting SDV Synthesizer...")
# Ensure no NaNs before fitting (SDV might handle some, but best to pre-fill)
df_sdv_filled = df_sdv.ffill().bfill()
synthesizer.fit(df_sdv_filled)

# --- 3. Generate Synthetic Sequences ---
# Decide how many sequences and their length
# Option A: Generate one long sequence
n_steps_total = 300 # Target total length
# Note: PAR might generate sequences of slightly varying lengths; you might need post-processing
synthetic_sequences = synthesizer.sample(num_sequences=1, sequence_length=n_steps_total)

# Option B: Generate multiple shorter sequences (e.g., if you had multiple entities)
# num_sequences_to_gen = 10
# sequence_length_avg = 150
# synthetic_sequences = synthesizer.sample(num_sequences=num_sequences_to_gen) # Let PAR decide length

print("\nSynthetic Data Sample (SDV based):")
print(synthetic_sequences.head())

# --- 4. Post-processing and Evaluation ---
# Set the date index back
synthetic_sequences_final = synthetic_sequences.set_index('date').sort_index()

# --- Evaluation ---
# Compare distributions, ACF, CCF, plots of synthetic_sequences_final vs df

target_to_plot = target_cols[0] # Choose one target to compare
plt.figure(figsize=(12, 6))
plt.plot(df.index, df[target_to_plot], label='Original Data')
plt.plot(synthetic_sequences_final.index, synthetic_sequences_final[target_to_plot], label='Synthetic Data (SDV)', alpha=0.7)
plt.title(f'Original vs. Synthetic Time Series ({target_to_plot}) - SDV')
plt.legend()
plt.show()

# Example Distribution Comparison:
plt.figure(figsize=(10, 5))
df[target_to_plot].hist(alpha=0.6, label='Original', bins=30, density=True)
synthetic_sequences_final[target_to_plot].hist(alpha=0.6, label='Synthetic (SDV)', bins=30, density=True)
plt.title(f'Distribution Comparison ({target_to_plot}) - SDV')
plt.legend()
plt.show()

# Save the synthesizer if needed
# synthesizer.save('my_par_synthesizer.pkl')
# loaded_synthesizer = PARSynthesizer.load('my_par_synthesizer.pkl')