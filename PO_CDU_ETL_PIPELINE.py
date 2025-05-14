import numpy as np
import pandas as pd
# import timesynth as ts # No longer needed for signal generation
import json
import re
import os
from datetime import datetime
import pickle

"""
Preprocesses extracted refinery data from a JSON file for input into a 
Temporal Fusion Transformer (TFT) model (Darts compatible format), following the guidelines.
Generates synthetic daily data for quality and yield targets based on averages.
Uses actual tag data for flows and the first 499 monitoring tags as past covariates.
Includes placeholders ONLY for Dosage data. Weather data is REMOVED.
Omits static covariates.
"""

# --- Configuration ---
base_path = "C:/Users/PC/Documents/DATABASE/PROC_OPTIM/Monthly Reports/JUNE,2024"
input_json_path = "extracted_refinery_data.json"
output_tft_data_path = f"{base_path}/tft_ready_data_darts.csv"
output_pickle_path = "extract_table.pkl"


# --- Utility Functions (safe_float, remove_nans) ---
# Keep these functions as they were

def safe_float(value):
    if pd.isna(value) or value in [None, '', 'nan', 'NaN', '-']: return np.nan
    try:
        if isinstance(value, str):
            value = value.strip().replace(',', '').replace('%', '')
            if not value: return np.nan
        return float(value)
    except (ValueError, TypeError):
        return np.nan


def remove_nans(obj):
    if isinstance(obj, dict):
        processed_dict = {k: remove_nans(v) for k, v in obj.items()}
        return {k: v for k, v in processed_dict.items() if v is not None}
    elif isinstance(obj, list):
        processed_list = [remove_nans(item) for item in obj]
        return [item for item in processed_list if item is not None]
    return obj


# --- Data Loading ---

def load_data(json_path):
    """Loads data from the extracted JSON file."""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    print(f"Loading data from {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)
    cleaned_data = remove_nans(data)  # Clean immediately
    print("Data loaded and cleaned successfully.")
    return cleaned_data


# --- Time Series Generation Utility ---
def generate_synthetic_target(base_value, std_dev_factor, days, clamp_buffer=0.01):
    """
    Generates a synthetic daily time series using only numpy noise around a base value.
    """
    try:
        base_value = float(base_value)
    except (ValueError, TypeError):
        print(f"Warning: Could not convert base_value '{base_value}' to float. Returning zeros.")
        return np.zeros(days)
    if base_value <= 0: return np.zeros(days)
    mean = base_value
    std_daily = abs(mean) * std_dev_factor if mean != 0 else 0.01
    noise_sample = np.random.normal(loc=0.0, scale=std_daily, size=days)
    synthetic_data = noise_sample + mean
    min_val = np.min(synthetic_data)
    offset = (-min_val + clamp_buffer) if min_val < 0 else 0
    return synthetic_data + offset


# --- Data Preparation for Darts TFT ---

def get_dates_and_tags_darts(ts_monitoring_data):
    """
    Extracts date range and reshapes tag data.
    Selects the first 499 unique tags found.
    Returns dates and a WIDE DataFrame of tags (date index, tag columns).
    """
    if not ts_monitoring_data or not isinstance(ts_monitoring_data, list):
        raise ValueError("Invalid or empty 'ts_monitoring' data provided in JSON.")
    example_entry = ts_monitoring_data[0]
    date_cols = []
    for k in example_entry.keys():
        if k != 'Tag':
            try:
                pd.to_datetime(k, errors='raise'); date_cols.append(k)
            except (ValueError, TypeError):
                continue
    if not date_cols: raise ValueError("Could not automatically identify date columns in 'ts_monitoring'.")
    print(f"Identified {len(date_cols)} date columns.")
    try:  # Attempt sorting
        date_cols.sort(key=lambda date: datetime.strptime(date, '%m/%d/%Y'))
        dates = pd.to_datetime(date_cols, format='%m/%d/%Y').sort_values()
    except ValueError:
        print("Warning: Could not sort date columns with format '%m/%d/%Y'. Using inferred format.")
        dates = pd.to_datetime(date_cols, infer_datetime_format=True).sort_values()

    df_ts = pd.DataFrame(ts_monitoring_data)
    if 'Tag' in df_ts.columns:
        df_ts['Tag'] = df_ts['Tag'].astype(str)
    else:
        raise ValueError("'Tag' column not found in 'ts_monitoring' data.")

    unique_tags = df_ts['Tag'].unique()
    if len(unique_tags) > 499:
        print(f"Found {len(unique_tags)} unique tags. Selecting the first 499.")
        selected_tags = unique_tags[:499]
        df_ts_filtered = df_ts[df_ts['Tag'].isin(selected_tags)].copy()
    else:
        print(f"Found {len(unique_tags)} unique tags (<= 499). Using all.")
        df_ts_filtered = df_ts.copy()

    df_long = pd.melt(df_ts_filtered, id_vars=['Tag'], value_vars=date_cols, var_name='date_str', value_name='value')
    df_long['date'] = pd.to_datetime(df_long['date_str'], errors='coerce')
    df_long = df_long.dropna(subset=['date'])
    df_long = df_long.drop(columns=['date_str'])
    df_long['value'] = pd.to_numeric(df_long['value'], errors='coerce')
    df_long = df_long.dropna(subset=['value'])

    try:
        df_wide_tags = df_long.pivot_table(index='date', columns='Tag', values='value')
    except Exception as e:
        print(f"Error pivoting tag data: {e}. Trying aggregation.")
        df_wide_tags = df_long.pivot_table(index='date', columns='Tag', values='value', aggfunc='mean')

    df_wide_tags = df_wide_tags.reindex(dates).ffill().bfill()
    print(f"Reshaped selected tags ({len(df_wide_tags.columns)}) to wide format. Shape: {df_wide_tags.shape}")
    return dates, df_wide_tags


def prepare_targets_darts(json_data, dates):
    """Prepares target variables (Quality, Yield - synthetic) in wide format."""
    print("\nPreparing target variables (Wide Format)...")
    df_targets_wide = pd.DataFrame(index=dates)
    num_days = len(dates)

    # --- 1. Quality Targets (Synthetic) ---
    quality_metrics = {}
    product_lab_data = json_data.get("product_lab_analysis", [])
    guide_quality_products = ["Naphtha", "Kero", "Combined Diesel", "RCO"]
    if isinstance(product_lab_data, list) and product_lab_data:
        print(f"Processing {len(product_lab_data)} entries in product_lab_analysis...")
        for entry in product_lab_data:
            metric_name = entry.get("Parameter")
            if isinstance(metric_name, str):
                metric_name_clean = re.sub(r'\s*\(.*\)\s*', '', metric_name).strip()
                for product_col_name in entry.keys():
                    if product_col_name in guide_quality_products:
                        value = entry.get(product_col_name)
                        float_value = safe_float(value)
                        if not np.isnan(float_value):
                            quality_metrics.setdefault(product_col_name, {})
                            quality_metrics[product_col_name][metric_name_clean] = float_value
    quality_target_count = 0
    for product, metrics in quality_metrics.items():
        for metric, mean_value in metrics.items():
            target_series = generate_synthetic_target(mean_value, 0.01, num_days)
            target_col_name = f"target_{product}_{metric}"
            df_targets_wide[target_col_name] = target_series
            quality_target_count += 1
    print(f"Generated {quality_target_count} synthetic quality target series.")

    # --- 2. Yield Targets (Synthetic based on CB Average) ---
    product_yield_avg = {}
    product_yield_data = json_data.get("product_yield", [])
    if isinstance(product_yield_data, list):
        cb_pattern = re.compile(r'^CB\d+( Rev)?$')
        guide_yield_products = ["LPG", "Naphtha", "Kero", "Light_Diesel", "Heavy_Diesel", "RCO"]
        for entry in product_yield_data:
            parameter = entry.get("Parameter");
            product_name = ""
            if isinstance(parameter, str) and "Yield %" not in parameter and "Total" not in parameter:
                product_name = parameter.strip()
            if isinstance(entry, dict):
                cb_values = [v for k, v in entry.items() if
                             cb_pattern.match(k) and isinstance(v, (int, float)) and not np.isnan(v)]
                if cb_values and product_name:
                    if "Naphtha" in product_name:
                        standard_name = "Naphtha"
                    elif "Kerosene" in product_name:
                        standard_name = "Kero"
                    elif "Light Diesel" in product_name:
                        standard_name = "Light_Diesel"
                    elif "Heavy Diesel" in product_name:
                        standard_name = "Heavy_Diesel"
                    else:
                        standard_name = product_name
                    if standard_name in guide_yield_products:
                        product_yield_avg[f"target_{standard_name}_Yield"] = np.mean(cb_values)
            else:
                print(f"Warning: Unexpected item type in product_yield data: {type(entry)}")
    yield_target_count = 0
    for target_col_name, mean_value in product_yield_avg.items():
        target_series = generate_synthetic_target(mean_value, 0.02, num_days)
        df_targets_wide[target_col_name] = target_series
        yield_target_count += 1
    print(f"Generated {yield_target_count} synthetic yield target series.")

    print(f"Targets DataFrame shape (before adding flows): {df_targets_wide.shape}")
    return df_targets_wide


def prepare_past_covariates_darts(df_wide_tags):
    """Prepares past covariates (Selected Tags ONLY)."""
    print("\nPreparing past covariates (Wide Format)...")
    # 1. Monitoring Tags (already wide)
    df_past_cov = df_wide_tags.copy()
    print(f"Using {len(df_past_cov.columns)} tags as past covariates.")
    # 2. REMOVED Weather Data Placeholder
    return df_past_cov


def prepare_future_covariates_darts(json_data, dates):
    """Prepares future covariates (Blend Info, Dosage) in wide format."""
    print("\nPreparing future covariates (Wide Format)...")
    num_days = len(dates)
    df_future_cov = pd.DataFrame(index=dates)

    # 1. Blend Info (API, Sulphur, Blend Number from CBxx)
    blend_comp_data = json_data.get("blend_composition", [])
    if isinstance(blend_comp_data, list) and blend_comp_data:
        print("  - Processing Blend Composition data...")
        cb_pattern = re.compile(r'^CB\d+( Rev)?$')
        blend_props = {'API': {}, 'Sulphur': {}}
        cb_columns_in_data = set()
        api_row = next((item for item in blend_comp_data if item.get("Parameter") == "API"), None)
        sulphur_row = next((item for item in blend_comp_data if item.get("Parameter") == "Sulphur"), None)
        if api_row:
            for k, v in api_row.items():
                if cb_pattern.match(k): blend_props['API'][k] = safe_float(v); cb_columns_in_data.add(k)
        if sulphur_row:
            for k, v in sulphur_row.items():
                if cb_pattern.match(k): blend_props['Sulphur'][k] = safe_float(v); cb_columns_in_data.add(k)
        cb_columns_ordered = sorted(list(cb_columns_in_data), key=lambda x: int(re.search(r'\d+', x).group()))

        if blend_props['API'] and blend_props['Sulphur'] and cb_columns_ordered:
            num_blends = len(cb_columns_ordered)
            print(f"    - Found {num_blends} CB columns with API/Sulphur data: {cb_columns_ordered}")
            blend_indices = [i % num_blends for i in range(num_days)]
            api_values = [blend_props['API'].get(cb_columns_ordered[i], np.nan) for i in blend_indices]
            sulphur_values = [blend_props['Sulphur'].get(cb_columns_ordered[i], np.nan) for i in blend_indices]
            blend_num_values = [int(re.search(r'\d+', cb_columns_ordered[i]).group()) for i in blend_indices]
            df_future_cov['future_API'] = api_values
            df_future_cov['future_Sulphur'] = sulphur_values
            df_future_cov['future_Blend_Num'] = blend_num_values
            df_future_cov['future_API'] = df_future_cov['future_API'].ffill().bfill()
            df_future_cov['future_Sulphur'] = df_future_cov['future_Sulphur'].ffill().bfill()
            print("    - Created future_API, future_Sulphur, future_Blend_Num columns.")
        else:
            print("    - Warning: Could not find sufficient API/Sulphur data per CB column.")
            df_future_cov['future_API'] = np.nan;
            df_future_cov['future_Sulphur'] = np.nan;
            df_future_cov['future_Blend_Num'] = np.nan
    else:
        print("  - Warning: 'blend_composition' data not found or invalid.")
        df_future_cov['future_API'] = np.nan;
        df_future_cov['future_Sulphur'] = np.nan;
        df_future_cov['future_Blend_Num'] = np.nan

    # 2. Dosage Data (Placeholder)
    print("  - Adding PLACEHOLDER future dosage data (Neutralizer, AntiCorrosion)...")
    df_future_cov['future_Neutralizer_dosage'] = 10 + np.random.normal(0, 1, num_days)
    df_future_cov['future_AntiCorrosion_dosage'] = 5 + np.random.normal(0, 0.5, num_days)

    return df_future_cov


# --- Main Darts Preprocessing Function ---

def preprocess_for_tft_darts(json_data):
    """Main function to preprocess data into Darts compatible format."""

    # 1. Extract dates and WIDE tag data (first 499 tags)
    dates, df_wide_tags = get_dates_and_tags_darts(json_data.get('ts_monitoring'))

    # 2. Prepare WIDE target variables (Quality, Yield - synthetic)
    df_targets_wide = prepare_targets_darts(json_data, dates)

    # --- Add Flowrate Targets (Actual) ---
    flow_tags_map = {
        "target_RCO_flow": "101FIC3101",
        "target_Heavy_Diesel_flow": "101FIC4201",
        "target_Light_Diesel_flow": "101FIC5303",
        "target_Kero_flow": "101FIC3001",
        "target_Naphtha_flow": "102FIC3201"
    }
    flow_targets_found = 0
    print("\nAdding Flowrate Targets...")
    for target_name, tag_name in flow_tags_map.items():
        if tag_name in df_wide_tags.columns:
            flow_series = df_wide_tags[tag_name].copy()
            min_val = flow_series.min()
            offset = (-min_val + 0.01) if min_val < 0 else 0
            df_targets_wide[target_name] = flow_series + offset
            flow_targets_found += 1
            print(f"  - Added flow target: {target_name} (from tag {tag_name})")
        else:
            print(
                f"  - Warning: Flow tag '{tag_name}' for '{target_name}' not found in selected monitoring tags. Target column will be missing.")
    print(f"Added {flow_targets_found} actual flowrate target series.")

    # 3. Prepare WIDE past covariates (Selected Tags ONLY - Weather Removed)
    df_past_cov_wide = prepare_past_covariates_darts(df_wide_tags)  # Pass only tags

    # 4. Prepare WIDE future covariates (Blend + Dosage)
    df_future_cov_wide = prepare_future_covariates_darts(json_data, dates)

    # 5. Combine into a single DataFrame
    print("\nCombining into final Darts DataFrame...")
    df_final_darts = df_targets_wide.copy()
    past_cov_cols_to_add = [col for col in df_past_cov_wide.columns if col not in df_final_darts.columns]
    df_final_darts = pd.concat([df_final_darts, df_past_cov_wide[past_cov_cols_to_add]], axis=1)
    df_final_darts = pd.concat([df_final_darts, df_future_cov_wide], axis=1)

    df_final_darts.index = dates
    time_delta = (df_final_darts.index - df_final_darts.index.min()).days
    df_final_darts['time_idx'] = time_delta

    print("Applying forward/backward fill for any remaining NaNs...")
    df_final_darts = df_final_darts.ffill().bfill()

    nan_check = df_final_darts.isnull().sum().sum()
    if nan_check > 0:
        print(f"\nWarning: {nan_check} NaNs remain in the final DataFrame!")
        print(df_final_darts.isnull().sum()[df_final_darts.isnull().sum() > 0])
    else:
        print("No NaNs found in the final DataFrame.")

    print(f"\nFinal Darts-ready DataFrame shape: {df_final_darts.shape}")
    print(
        f"Final Columns (first 10 and last 5): {df_final_darts.columns[:10].tolist()} ... {df_final_darts.columns[-5:].tolist()}")
    return df_final_darts


# --- Execution ---
if __name__ == "__main__":
    # 1. Load extracted data
    try:
        extracted_data = load_data(input_json_path)
        if not extracted_data:
            print("Error: Loaded JSON data is empty.")
            exit()
    except FileNotFoundError as e:
        print(e); exit()
    except Exception as e:
        print(f"Error loading JSON data: {e}"); exit()

    # 2. Preprocess for Darts TFT
    print("-" * 20)
    print("Preprocessing data for Darts TFT...")
    print("-" * 20)
    try:
        darts_ready_df = preprocess_for_tft_darts(extracted_data)

        # 3. Save the preprocessed data
        print(f"\nSaving Darts TFT-ready data to {output_tft_data_path}...")
        darts_ready_df.to_csv(output_tft_data_path, index=True)  # Keep date index
        print("Darts TFT-ready data saved successfully.")
        print("\nSample of final data:")
        print(darts_ready_df.head())

    except ValueError as ve:
        print(f"\nValueError during preprocessing: {ve}")
    except Exception as e:
        print(f"\nAn unexpected error occurred during preprocessing: {e}")
        import traceback

        traceback.print_exc()

    # 4. Serialize the extract_table function (Optional - Needs function definition)
    # if 'extract_table' in globals():
    #     try:
    #         print(f"\nPickling the 'extract_table' function to: {output_pickle_path}")
    #         with open(output_pickle_path, "wb") as f: pickle.dump(extract_table, f)
    #         print("'extract_table' function pickled successfully.")
    #     except Exception as e: print(f"Error pickling function: {e}.")

    print("-" * 20)
    print("Script finished.")
    print("-" * 20)
