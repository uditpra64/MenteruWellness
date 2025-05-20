# optimization_module/optimize_utils.py
import logging
import os # For os.path.join and os.makedirs
import numpy as np
import pandas as pd
from data_module.config.utils import get_path_from_config # Only needed for df_for_normalize_out_path

# Ensure these are correctly imported relative to this file's location or project structure
from prediction_module.electricity_module.prediction_logic import run_prediction_logic
from prediction_module.utilities.models import XGBoostModel
from prediction_module.wellness_module.comfort_logic import calculate_comfort
from prediction_module.wellness_module.productivity_logic import calculate_productivity
from tqdm import tqdm

logger = logging.getLogger(__name__)

def create_df_for_normalize(
    df: pd.DataFrame,
    model: XGBoostModel,
    lineage: str,
    # THIS IS THE CRITICAL ARGUMENT: optimization_params_sheet
    # It's the "最適化" sheet DataFrame, already correctly read by OptimizationRunner
    optimization_params_sheet: pd.DataFrame, 
    start_study_date: str = None,
    end_study_date: str = None,
    input_features_columns: list[str] = None,
    temperature_setpoints_column_name: str = None, 
) -> pd.DataFrame:
    df_copy = df.copy()
    df_copy["datetime"] = pd.to_datetime(df_copy["datetime"])
    df_copy["date"] = pd.to_datetime(df_copy["date"])
    df_copy = pd.get_dummies(df_copy, dtype=int, columns=['DayType'], prefix='DayType', dummy_na=False) # Ensure DayType is handled

    if start_study_date and end_study_date:
        df_copy = df_copy[
            (df_copy["date"] >= pd.to_datetime(start_study_date)) & 
            (df_copy["date"] <= pd.to_datetime(end_study_date))
        ]
    df_copy = df_copy.reset_index(drop=True)

    data_for_normalize = []
    lineage_filename_part = lineage.replace("_", "")

    # These are the TARGET COLUMN NAMES we expect in `optimization_params_sheet`
    # (which becomes `extracted_master_data` after filtering by month)
    # These names come from Excel Row 2, as read by OptimizationRunner with header=1
    expected_productivity_coeffs = [
        "知的生産性_朝_A", "知的生産性_朝_B", "知的生産性_朝_C",
        "知的生産性_昼_A", "知的生産性_昼_B", "知的生産性_昼_C",
        "知的生産性_夕_A", "知的生産性_夕_B", "知的生産性_夕_C"
    ]
    # From pandas debug output of "最適化" sheet (read with header=1 by Runner)
    expected_comfort_params = [
        '室内相対湿度 [%] ', # Note the trailing space!
        '代謝量 [met]',
        '着衣量 [clo]',
        '気流速度 [m/s]'
    ]
    
    # Initialize prev_hour_setpoint_for_pmv
    # Use a typical default if df_copy is empty or column is missing
    if not df_copy.empty and temperature_setpoints_column_name in df_copy.columns:
        prev_hour_setpoint_for_pmv = df_copy[temperature_setpoints_column_name].iloc[0]
    else:
        prev_hour_setpoint_for_pmv = 24.0 # Default setpoint

    for time_idx in tqdm(
        range(len(df_copy)), desc=f"df_for_normalize_{lineage_filename_part}を作成中", ncols=100
    ):
        # Use the optimization_params_sheet that was PASSED IN.
        # DO NOT re-read the Excel file here.
        extracted_master_data = extract_master_data(df_copy, time_idx, optimization_params_sheet) 

        # --- DEBUG PRINT (Optional: Keep for one run if errors persist on these columns) ---
        # print(f"--- DEBUG create_df_for_normalize (time_idx: {time_idx}, month: {df_copy['date'][time_idx].month}) ---")
        # print("Columns in extracted_master_data (create_df_for_normalize):", extracted_master_data.columns.tolist())
        # print("Content of extracted_master_data (create_df_for_normalize):\n", extracted_master_data.to_string())
        # print("------------------------------------------------------------------------------------")
        # --- END DEBUG PRINT ---

        missing_pro_cols = [col for col in expected_productivity_coeffs if col not in extracted_master_data.columns]
        if missing_pro_cols:
            raise KeyError(f"From create_df_for_normalize: Missing productivity coefficient columns in extracted_master_data: {missing_pro_cols}. Available: {extracted_master_data.columns.tolist()}")
        master_pro_coeffs = extracted_master_data.iloc[0:1][expected_productivity_coeffs].copy()

        missing_com_cols = [col for col in expected_comfort_params if col not in extracted_master_data.columns]
        if missing_com_cols:
             # This is where the current error in your log is coming from.
             raise KeyError(f"From create_df_for_normalize: Missing comfort parameter columns in extracted_master_data: {missing_com_cols}. Expected: {expected_comfort_params}. Available: {extracted_master_data.columns.tolist()}")
        master_com_params = extracted_master_data.iloc[0:1][expected_comfort_params].copy()
        
        current_row_df = df_copy.iloc[time_idx:time_idx+1].copy()

        # Ensure the setpoint column exists in current_row_df
        if temperature_setpoints_column_name not in current_row_df.columns:
            raise KeyError(f"Setpoint column '{temperature_setpoints_column_name}' not found in current_row_df during create_df_for_normalize. Available: {current_row_df.columns.tolist()}")

        energy_consumption = run_prediction_logic(current_row_df, input_features_columns, 0, model)
        productivity_value = calculate_productivity(current_row_df, temperature_setpoints_column_name, master_pro_coeffs)
        
        if time_idx > 0 and temperature_setpoints_column_name in df_copy.columns:
             prev_hour_setpoint_for_pmv = df_copy[temperature_setpoints_column_name].iloc[time_idx-1]
        # else, it keeps its initial value or previous iteration's value. This ensures it's always defined.

        comfort_value, _, _ = calculate_comfort(current_row_df, temperature_setpoints_column_name, master_com_params, prev_hour_setpoint_for_pmv)
        comfort_value_exp = np.exp(0.05 * comfort_value) if pd.notna(comfort_value) else np.nan

        data_for_normalize.append({
            "datetime": df_copy.loc[time_idx, "datetime"],
            "消費電力量": energy_consumption,
            "快適性指標": comfort_value,
            "快適性指標_exp": comfort_value_exp,
            "知的生産性": productivity_value,
        })
        
        if temperature_setpoints_column_name in current_row_df.columns: # Update for next iter if col exists
            prev_hour_setpoint_for_pmv = current_row_df[temperature_setpoints_column_name].iloc[0]


    df_for_normalize_out = pd.DataFrame(data_for_normalize)
    # Construct path for saving df_for_normalize
    df_for_normalize_dir = get_path_from_config("df_for_normalize_path")
    os.makedirs(df_for_normalize_dir, exist_ok=True) # Ensure directory exists
    df_for_normalize_out_path = os.path.join(
        df_for_normalize_dir,
        f"df_for_normalize_{lineage_filename_part}.csv"
    )
    df_for_normalize_out.to_csv(df_for_normalize_out_path, index=False, encoding="cp932")
    logger.info(f"df_for_normalize out path: {df_for_normalize_out_path}")
    return df_for_normalize_out

def normalize_data(data_value, normalization_series: pd.Series):
    if pd.isna(data_value): # Handle NaN input directly
        return np.nan 
    if isinstance(data_value, (int, float, np.number)):
        min_val = normalization_series.min()
        max_val = normalization_series.max()
        if pd.isna(min_val) or pd.isna(max_val) or (max_val - min_val) == 0:
            return 0.0 
        return (data_value - min_val) / (max_val - min_val)
    elif isinstance(data_value, pd.Series): # Should not happen with current calls
        min_val = normalization_series.min()
        max_val = normalization_series.max()
        if pd.isna(min_val) or pd.isna(max_val) or (max_val - min_val) == 0:
            return pd.Series([0.0] * len(data_value), index=data_value.index)
        return (data_value - min_val) / (max_val - min_val)
    else:
        logger.error(f"Unsupported types for normalize_data: data_value is {type(data_value)}, normalization_series is {type(normalization_series)}")
        return np.nan

def extract_master_data(
    df_current_timestep_info: pd.DataFrame,
    time_idx_for_df: int, # This is the index for df_current_timestep_info (usually 0)
    master_data_optim_sheet_full: pd.DataFrame # This IS the "最適化" sheet from runner
) -> pd.DataFrame:
    current_month = df_current_timestep_info["date"].iloc[time_idx_for_df].month
    
    if master_data_optim_sheet_full.empty:
        raise ValueError("extract_master_data received an empty master_data_optim_sheet_full.")

    # "最適化" sheet (master_data_optim_sheet_full) as read by runner has 'Unnamed: 0' 
    # as its first column, which contains the month number.
    month_col_name_in_sheet = master_data_optim_sheet_full.columns[0] 
    
    try:
        numeric_month_values_in_sheet = pd.to_numeric(master_data_optim_sheet_full[month_col_name_in_sheet], errors='coerce')
    except Exception as e:
        raise TypeError(f"Could not process month column ('{month_col_name_in_sheet}') in '最適化' sheet for numeric conversion. Error: {e}.")

    monthly_data_row_df = master_data_optim_sheet_full[numeric_month_values_in_sheet == current_month].copy()

    if monthly_data_row_df.empty:
        unique_months_in_sheet_coerced = numeric_month_values_in_sheet.dropna().unique()
        original_month_col_values = master_data_optim_sheet_full[month_col_name_in_sheet].unique()
        raise ValueError(
            f"No master data found for month: {current_month} using column '{month_col_name_in_sheet}' in '最適化' sheet. "
            f"Available numeric months after coercion: {unique_months_in_sheet_coerced}. "
            f"Original unique values in sheet's month column '{month_col_name_in_sheet}': {original_month_col_values}"
        )
    
    return monthly_data_row_df.iloc[0:1, :].reset_index(drop=True)