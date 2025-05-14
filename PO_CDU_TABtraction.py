import pandas as pd
import json
import pickle
import math
import numpy as np
import os
import re  # Import regular expressions for finding CBxx columns

"""This script extracts structured tables from the provided CSV files (Lab.csv, TS Monitoring Tags.csv, blend.csv), 
removes NaN values, and converts them into JSON format."""

# --- Configuration ---
# Base path for input files
base_path = "C:/Users/PC/Documents/DATABASE/PROC_OPTIM/Monthly Reports/JUNE,2024"
# Output file paths
output_json_path = "extracted_refinery_data.json"
output_pickle_path = "extract_table.pkl"

# Dictionary mapping logical names to actual CSV file paths
csv_files = {
    'lab_data': os.path.join(base_path, "Lab.csv"),
    'ts_monitoring': os.path.join(base_path, "TS Monitoring Tags.csv"),
    'blend_data': os.path.join(base_path, "blend.csv")
}


# --- Utility Functions ---

def safe_float(value):
    """
    Safely convert a value to float, handling common non-numeric representations and percentages.
    Returns np.nan if conversion fails or input is considered empty/null.
    """
    if pd.isna(value) or value in [None, '', 'nan', 'NaN', '-']:
        return np.nan
    try:
        if isinstance(value, str):
            # Remove common artifacts like commas, percentage signs, and whitespace
            value = value.strip().replace(',', '').replace('%', '')
            if not value:  # Handle case where string becomes empty after stripping
                return np.nan
        return float(value)
    except (ValueError, TypeError):
        # Return NaN if conversion still fails
        return np.nan


def remove_nans(obj):
    """
    Recursively remove None values (representing original NaNs) from nested structures.
    """
    if isinstance(obj, dict):
        processed_dict = {k: remove_nans(v) for k, v in obj.items()}
        return {k: v for k, v in processed_dict.items() if v is not None}
    elif isinstance(obj, list):
        processed_list = [remove_nans(item) for item in obj]
        return [item for item in processed_list if item is not None]
    return obj


def extract_table(csv_path, table_name, skiprows=None, header_row=None, columns=None, usecols=None, nrows=None):
    """
    Extract a table from a CSV file with flexible header and column handling.
    (Modified to better handle assigning columns when header_row is None)
    """
    print(f"\nAttempting to extract table '{table_name}' from: {csv_path}")
    print(
        f"Parameters: skiprows={skiprows}, header_row={header_row}, columns={columns is not None}, usecols={usecols}, nrows={nrows}")

    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found: {csv_path}")
        return None

    try:
        read_header = header_row if columns is None else None
        df = pd.read_csv(
            csv_path,
            skiprows=skiprows,
            header=read_header,
            usecols=usecols,
            nrows=nrows,
            encoding='utf-8',
            keep_default_na=True
        )
        print(f"Read {len(df)} rows initially for '{table_name}'.")
        initial_cols = df.columns.tolist()

        if columns:
            if len(initial_cols) >= len(columns):
                df = df.iloc[:, :len(columns)]
                df.columns = columns
                print(f"Assigned custom columns for '{table_name}': {columns}")
            else:
                print(
                    f"Warning: Read {len(initial_cols)} columns, but {len(columns)} column names provided for '{table_name}'. Cannot assign names. Keeping default.")
                df.columns = [str(col).strip() for col in df.columns]
        elif header_row is not None:
            df.columns = [str(col).strip() if pd.notna(col) else f'Unnamed: {i}' for i, col in enumerate(df.columns)]
            print(f"Cleaned columns read from header for '{table_name}': {df.columns.tolist()}")
        else:
            df.columns = [str(col).strip() for col in df.columns]

        df = df.dropna(axis=0, how='all')
        print(f"Rows after dropping all-NaN rows for '{table_name}': {len(df)}")

        # Apply safe_float conversion - ONLY apply to columns that are NOT the parameter column
        parameter_col_name = df.columns[0]  # Assume first column is the parameter name
        for col in df.columns:
            if col != parameter_col_name:  # Skip the parameter column
                converted_col = []
                for _, value in df[col].items():
                    float_val = safe_float(value)
                    # Store the float if conversion successful, otherwise keep original (or None if original was NaN)
                    converted_col.append(
                        float_val if not np.isnan(float_val) else (value if not pd.isna(value) else None))
                df[col] = converted_col  # Update column in DataFrame
            else:
                # Ensure parameter column is string and stripped
                df[col] = df[col].astype(str).str.strip()

        df = df.replace({np.nan: None})
        df = df.reset_index(drop=True)

        result = df.to_dict(orient='records')
        print(f"Successfully extracted and processed {len(result)} records for '{table_name}'.")
        return result

    except Exception as e:
        print(f"Error extracting table '{table_name}' from {csv_path}: {e}")
        import traceback
        traceback.print_exc()
        return None


# --- Specific Table Extraction Logic ---

# Keep extract_lab_tables and extract_ts_monitoring_table as they are

def extract_lab_tables():
    """Extract tables from Lab.csv (Simplified)."""
    csv_path = csv_files['lab_data']
    if not os.path.exists(csv_path):
        print(f"Error: Lab CSV file not found at {csv_path}")
        return {'raw_crude_lab_analysis': None, 'product_lab_analysis': None, 'overhead_water_analysis': None}

    tables = {
        'raw_crude_lab_analysis': None,
        'product_lab_analysis': None,
        'overhead_water_analysis': None
    }

    # --- Extract Raw Crude Lab Analysis ---
    crude_columns_to_extract = ['Parameter', 'Limit', 'Raw Crude', 'Train A', 'Train B']
    crude_usecols_indices = [0, 2, 4, 5, 6]  # Indices for Parameter, Limit, Raw Crude, Train A, Train B

    tables['raw_crude_lab_analysis'] = extract_table(
        csv_path=csv_path,
        table_name='raw_crude_lab_analysis',
        skiprows=3,  # Skip title, blank, header row
        header_row=None,  # We are assigning columns manually
        columns=crude_columns_to_extract,  # Assign these names
        usecols=crude_usecols_indices,  # Read only these columns by index
        nrows=7  # Expected number of data rows for this section
    )

    # --- Extract Product Lab Analysis ---
    prod_columns_to_extract = ['Parameter', 'Naphtha', 'Kero', 'ATF', 'Light Diesel', 'Heavy Diesel', 'RCO',
                               'Combined Diesel']
    prod_usecols_indices = [0, 2, 3, 4, 5, 6, 7, 8]

    tables['product_lab_analysis'] = extract_table(
        csv_path=csv_path,
        table_name='product_lab_analysis',
        skiprows=11,  # Skip title, blank, header row for this section
        header_row=None,
        columns=prod_columns_to_extract,
        usecols=prod_usecols_indices,
        nrows=8  # Expected data rows for this section
    )

    # --- Extract Column Overhead Water Analysis ---
    water_columns_to_extract = ['Parameter', 'Limit', 'Value']
    water_usecols_indices = [0, 1, 2]

    tables['overhead_water_analysis'] = extract_table(
        csv_path=csv_path,
        table_name='overhead_water_analysis',
        skiprows=20,  # Skip previous sections and this section's header
        header_row=None,
        columns=water_columns_to_extract,
        usecols=water_usecols_indices,
        nrows=4  # Expected data rows for this section
    )

    return tables


def extract_ts_monitoring_table():
    """Extract time series monitoring data from TS Monitoring Tags.csv."""
    csv_path = csv_files['ts_monitoring']
    if not os.path.exists(csv_path):
        print(f"Error: TS Monitoring CSV file not found at {csv_path}")
        return None

    try:
        # Read header to determine date columns dynamically
        header_df = pd.read_csv(csv_path, skiprows=1, nrows=1, encoding='utf-8')
        all_cols = [str(col).strip() for col in header_df.columns]
        fixed_cols = ['S/N', 'DCS Tag', 'Tag', 'Average']
        date_cols = [col for col in all_cols if
                     col not in fixed_cols and col and not col.startswith('Unnamed:')]  # Filter out unnamed
        usecols_ts = ['Tag'] + date_cols  # Select 'Tag' and the identified date columns

        return extract_table(
            csv_path=csv_path,
            table_name='ts_monitoring',
            skiprows=1,
            header_row=0,  # Header is the first row after skipping
            usecols=usecols_ts,  # Use only Tag and Date columns
            columns=None  # Use header row for column names
        )
    except Exception as e:
        print(f"Error pre-processing or extracting 'ts_monitoring': {e}")
        return None


# --- MODIFIED extract_blend_table ---
def extract_blend_table():
    """Extract blend composition and product yield data for all CBxx columns from blend.csv."""
    csv_path = csv_files['blend_data']
    if not os.path.exists(csv_path):
        print(f"Error: Blend CSV file not found at {csv_path}")
        return {'blend_composition': None, 'product_yield': None}

    print(f"\nAttempting to extract full blend table from: {csv_path}")
    try:
        # Read the header row to identify columns
        header_df = pd.read_csv(csv_path, nrows=1, encoding='utf-8')
        all_columns = [str(col).strip() for col in header_df.columns]

        # Identify the parameter column (likely the first one)
        parameter_col_name = all_columns[0]
        # Identify 'CBxx' columns using regex (allowing for 'CBxx Rev')
        cb_pattern = re.compile(r'^CB\d+( Rev)?$')
        cb_columns = [col for col in all_columns if cb_pattern.match(col)]
        print(f"Identified Parameter Column: '{parameter_col_name}'")
        print(f"Identified CB Columns: {cb_columns}")

        # Read the full data using the identified header row (row 0)
        df_blend = pd.read_csv(csv_path, header=0, encoding='utf-8', keep_default_na=True)
        df_blend.columns = [str(col).strip() for col in df_blend.columns]  # Clean column names again

        # Process the DataFrame
        blend_composition = []
        product_yield = []
        in_yield_section = False

        for index, row in df_blend.iterrows():
            parameter = row[parameter_col_name]
            if pd.isna(parameter):
                continue  # Skip rows without a parameter name

            parameter_str = str(parameter).strip()

            if "Yield %" in parameter_str:
                in_yield_section = True
                continue  # Skip the section header row itself
            elif parameter_str == '':  # Skip blank rows
                continue

            # Start building the record for this parameter
            record = {'Parameter': parameter_str}
            has_valid_cb_value = False

            # Iterate through the identified CB columns
            for cb_col in cb_columns:
                if cb_col in row:  # Check if column exists in the row Series
                    cb_value_raw = row[cb_col]
                    cb_value_float = safe_float(cb_value_raw)
                    # Only add the CB value if it's a valid number (not NaN)
                    if not np.isnan(cb_value_float):
                        record[cb_col] = cb_value_float
                        has_valid_cb_value = True  # Mark that we found at least one number

            # Only add the record if it has a valid parameter AND at least one valid CB value
            if has_valid_cb_value:
                if in_yield_section:
                    if 'Total' not in parameter_str:  # Exclude Total row
                        product_yield.append(record)
                else:
                    # Check if parameter looks like a Crude name or API/Sulphur
                    if any(crude in parameter_str for crude in
                           ['AGB', 'BOL', 'CJB', 'WCD', 'QUI', 'AME', 'FOR', 'ABO', 'OK2', 'ESC', 'BOC', 'BRL', 'AKP',
                            'ASC', 'AGO', 'YOH', 'OKW', 'ERH', 'PAZ', 'Nembe', 'Utapate']) or parameter_str in ['API',
                                                                                                                'Sulphur']:
                        blend_composition.append(record)
            else:
                print(f"Skipping row for Parameter '{parameter_str}' - no valid CB values found.")

        print(f"Processed {len(blend_composition)} blend composition records.")
        print(f"Processed {len(product_yield)} product yield records.")
        return {'blend_composition': blend_composition, 'product_yield': product_yield}

    except Exception as e:
        print(f"Error processing blend table from {csv_path}: {e}")
        import traceback
        traceback.print_exc()
        return {'blend_composition': None, 'product_yield': None}


# --- Main Orchestration Function (Modified to use updated blend extraction) ---

def extract_refinery_csv_to_json():
    """Extract tables from CSV files and convert to JSON."""
    data = {}

    # Extract Lab tables
    lab_tables = extract_lab_tables()
    data.update(lab_tables)  # Add all keys from lab_tables

    # Extract TS Monitoring table
    data['ts_monitoring'] = extract_ts_monitoring_table()

    # Extract Blend and Yield tables (uses the MODIFIED function)
    blend_yield_tables = extract_blend_table()
    data['blend_composition'] = blend_yield_tables['blend_composition']
    data['product_yield'] = blend_yield_tables['product_yield']

    # --- Report Empty Tables (No Prompting) ---
    final_data = {}
    for key, value in data.items():
        if value is None or (isinstance(value, list) and not value):
            print(f"\nWarning: Table '{key}' was extracted as empty or failed to extract.")
            final_data[key] = value
        else:
            final_data[key] = value

    # Remove top-level keys with None or empty list values
    final_data = {k: v for k, v in final_data.items() if not (v is None or (isinstance(v, list) and not v))}

    # Clean remaining NaN/None values recursively
    cleaned_data = remove_nans(final_data)

    return cleaned_data


# --- Execution (Keep as is) ---
def main():
    print("-" * 20)
    print("Starting CSV Data Extraction")
    print("-" * 20)

    # Execute the extraction
    json_output = extract_refinery_csv_to_json()

    print("-" * 20)
    print("Extraction Summary (Tables included in JSON):")
    if json_output:
        for key in json_output.keys():
            print(f"- {key}")
    else:
        print("No data extracted or all extracted tables were empty.")
    print("-" * 20)

    # --- Output ---

    # Save JSON to a file
    try:
        print(f"Saving JSON output to: {output_json_path}")
        with open(output_json_path, "w") as f:
            json.dump(json_output, f, indent=2, default=str)
        print("JSON file saved successfully.")
    except Exception as e:
        print(f"Error saving JSON file: {e}")

    # Serialize the extract_table function (if useful)
    try:
        print(f"\nPickling the 'extract_table' function to: {output_pickle_path}")
        with open(output_pickle_path, "wb") as f:
            pickle.dump(extract_table, f)
        print("'extract_table' function pickled successfully.")
    except Exception as e:
        print(f"Error pickling function: {e}. Pickling functions can be problematic.")

    print("-" * 20)
    print("Script finished.")
    print("-" * 20)


if __name__ == "__main__":
    main()
