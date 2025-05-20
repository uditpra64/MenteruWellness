# optimization_module/optimize_logic.py
import logging
import time
import os
from typing import Tuple

import numpy as np
import pandas as pd
from config_settings.config_optimize import FINAL_OPTIMIZE_OUTOUT_COLUMN
from data_module.config.utils import (
    get_path_from_config,
    get_path_from_config_for_outside_base_folder,
)
from optimization_module.optimize_utils import (
    create_df_for_normalize,
    extract_master_data,
    normalize_data,
)
from prediction_module.electricity_module.prediction_logic import run_prediction_logic
try:
    from prediction_module.utilities.models import XGBoostModel
except ImportError:
    XGBoostModel = object 
from prediction_module.wellness_module.comfort_logic import calculate_comfort
from prediction_module.wellness_module.productivity_logic import calculate_productivity
from pythermalcomfort.utilities import v_relative
from tqdm import tqdm

logger = logging.getLogger(__name__)

def extract_master_data_values(
    df_current_timestep: pd.DataFrame, 
    conduct_time_idx_in_df: int, 
    optimization_params_sheet: pd.DataFrame, 
    case_weights_sheet: pd.DataFrame
) -> Tuple:
    month = df_current_timestep["date"].iloc[conduct_time_idx_in_df].month 
    extracted_master_data_row = extract_master_data(df_current_timestep, conduct_time_idx_in_df, optimization_params_sheet)

    month_col_in_case_sheet = case_weights_sheet.columns[0] 
    monthly_weights_row = case_weights_sheet[pd.to_numeric(case_weights_sheet[month_col_in_case_sheet], errors='coerce') == month]
    if monthly_weights_row.empty:
        available_months = case_weights_sheet[month_col_in_case_sheet].unique()
        raise ValueError(f"No weights for month {month} in '{getattr(case_weights_sheet, 'name', 'case_sheet')}'. Avail: {available_months}")
    latest_weight = monthly_weights_row.iloc[0:1].reset_index(drop=True)

    prod_coeff_cols = ["知的生産性_朝_A", "知的生産性_朝_B", "知的生産性_朝_C", "知的生産性_昼_A", "知的生産性_昼_B", "知的生産性_昼_C", "知的生産性_夕_A", "知的生産性_夕_B", "知的生産性_夕_C"]
    if not all(col in extracted_master_data_row.columns for col in prod_coeff_cols):
        raise KeyError(f"extract_master_data_values: Missing productivity coefficient columns. Needed: {prod_coeff_cols}. Got: {extracted_master_data_row.columns.tolist()}")
    master_pro = extracted_master_data_row.iloc[0:1][prod_coeff_cols].copy()

    comfort_param_cols = ['室内相対湿度 [%] ', '代謝量 [met]', '着衣量 [clo]', '気流速度 [m/s]'] # Trailing space on humidity is important
    if not all(col in extracted_master_data_row.columns for col in comfort_param_cols):
        raise KeyError(f"extract_master_data_values: Missing comfort parameter columns. Needed: {comfort_param_cols}. Got: {extracted_master_data_row.columns.tolist()}")
    master_com = extracted_master_data_row.iloc[0:1][comfort_param_cols].copy()

    overall_temp_range_max = extracted_master_data_row['上限値'].iloc[0]
    overall_temp_range_min = extracted_master_data_row['下限値'].iloc[0]
    
    # Corrected access for benchmark_temp based on DEBUG RUNNER output
    actual_benchmark_temp_col_name = 'Unnamed: 25' # This is from your DEBUG RUNNER output
    if actual_benchmark_temp_col_name not in extracted_master_data_row.columns:
        # Fallback or more descriptive error if "Unnamed: 25" is also not there or changes
        alt_benchmark_name = 'ベンチマークケース設定温度' # The ideal name from your "2nd row" list
        if alt_benchmark_name in extracted_master_data_row.columns:
            actual_benchmark_temp_col_name = alt_benchmark_name
            logger.warning(f"Using '{alt_benchmark_name}' for benchmark temperature as '{actual_benchmark_temp_col_name}' was not found initially.")
        else:
            raise KeyError(f"Column '{actual_benchmark_temp_col_name}' (or '{alt_benchmark_name}') expected for benchmark temperature not found in extracted_master_data_row. Available columns: {extracted_master_data_row.columns.tolist()}")
    benchmark_temp = extracted_master_data_row[actual_benchmark_temp_col_name].iloc[0]

    # Wellness component weights from "最適化" sheet (Excel columns V & W in image_1c0931.png)
    # Your "DEBUG RUNNER" output showed these are named '知的生産性' and '快適性指数'
    weight_pro_col_name = "知的生産性" 
    weight_com_col_name = "快適性指数" 

    if weight_pro_col_name in extracted_master_data_row.columns:
        weight_pro = extracted_master_data_row[weight_pro_col_name].iloc[0]
    else:
        logger.warning(f"Column '{weight_pro_col_name}' for prod. weight not found in '最適化' sheet (month {month}). Default 0.5. Avail: {extracted_master_data_row.columns.tolist()}")
        weight_pro = 0.5 
    if weight_com_col_name in extracted_master_data_row.columns:
        weight_com = extracted_master_data_row[weight_com_col_name].iloc[0]
    else:
        logger.warning(f"Column '{weight_com_col_name}' for comf. weight not found in '最適化' sheet (month {month}). Default 0.5. Avail: {extracted_master_data_row.columns.tolist()}")
        weight_com = 0.5
    
    prod_limit_cols = ["知的生産性_下限温度_朝", "知的生産性_上限温度_朝", "知的生産性_下限温度_昼", "知的生産性_上限温度_昼", "知的生産性_下限温度_夕", "知的生産性_上限温度_夕"]
    if not all(col in extracted_master_data_row.columns for col in prod_limit_cols):
        raise KeyError(f"extract_master_data_values: Missing productivity temperature limit columns. Needed: {prod_limit_cols}. Got: {extracted_master_data_row.columns.tolist()}")
    temp_range_morning_min = extracted_master_data_row["知的生産性_下限温度_朝"].iloc[0]
    temp_range_morning_max = extracted_master_data_row["知的生産性_上限温度_朝"].iloc[0]
    temp_range_afternoon_min = extracted_master_data_row["知的生産性_下限温度_昼"].iloc[0]
    temp_range_afternoon_max = extracted_master_data_row["知的生産性_上限温度_昼"].iloc[0]
    temp_range_evening_min = extracted_master_data_row["知的生産性_下限温度_夕"].iloc[0]
    temp_range_evening_max = extracted_master_data_row["知的生産性_上限温度_夕"].iloc[0]

    period_type_col_name = extracted_master_data_row.columns[1] # 'Unnamed: 1' for 期間区分
    period_type = extracted_master_data_row[period_type_col_name].iloc[0]

    return (
        month, latest_weight, master_pro, master_com, None, 
        weight_pro, weight_com, benchmark_temp,
        temp_range_morning_min, temp_range_morning_max,
        temp_range_afternoon_min, temp_range_afternoon_max,
        temp_range_evening_min, temp_range_evening_max,
        overall_temp_range_min, overall_temp_range_max,
        period_type
    )

def run_optimization_logic(
    df: pd.DataFrame,
    input_features_columns: list[str],
    temperature_setpoints_columns_dict: dict,
    lineage: str,
    start_study_date: str,
    end_study_date: str,
    start_optimize: str,
    end_optimize: str,
    model: XGBoostModel,
    master_data_excel_sheets: dict, 
    case_number: int,
    case_sheet_name: str,
    train_memory_flag: bool = False,
) -> pd.DataFrame:
    start_time_opt_logic = time.time()
    lineage_for_filename = lineage.replace("_", "")
    
    if lineage not in temperature_setpoints_columns_dict:
        raise KeyError(f"Setpoint column for lineage '{lineage}' not found in temperature_setpoints_columns_dict.")
    current_setpoint_column_name = temperature_setpoints_columns_dict[lineage]

    optimization_params_sheet = master_data_excel_sheets["最適化"]
    case_weights_sheet = master_data_excel_sheets[case_sheet_name]

    # df_for_normalize generation
    if not train_memory_flag:
        logger.info(f"Generating df_for_normalize for {lineage}...")
        df_for_normalize = create_df_for_normalize(
            df.copy(), model, lineage,
            optimization_params_sheet, 
            start_study_date, end_study_date,
            input_features_columns, current_setpoint_column_name,
        )
    else: 
        logger.info(f"train_memory_flag is True. Attempting to load df_for_normalize for {lineage}...")
        df_for_normalize_path_key = "df_for_normalize_path"
        df_for_normalize_base_path = get_path_from_config(df_for_normalize_path_key)
        file_to_load = os.path.join(df_for_normalize_base_path, f"df_for_normalize_{lineage_for_filename}.csv")
        try:
            df_for_normalize = pd.read_csv(file_to_load, encoding="cp932")
            logger.info(f"Successfully loaded {file_to_load}")
        except FileNotFoundError:
            logger.warning(f"{file_to_load} not found. Generating it now.")
            df_for_normalize = create_df_for_normalize(
                df.copy(), model, lineage,
                optimization_params_sheet,
                start_study_date, end_study_date,
                input_features_columns, current_setpoint_column_name,
            )
    
    if df_for_normalize.empty:
        logger.error(f"df_for_normalize is empty for lineage {lineage}. Cannot proceed with optimization.")
        return pd.DataFrame(columns=FINAL_OPTIMIZE_OUTOUT_COLUMN)


    df_opt_period = df.copy()
    df_opt_period["datetime"] = pd.to_datetime(df_opt_period["datetime"])
    df_opt_period["date"] = pd.to_datetime(df_opt_period["date"])
    if 'DayType' in df_opt_period.columns: # Check if DayType column exists
        df_opt_period = pd.get_dummies(df_opt_period, dtype=int, columns=['DayType'], prefix='DayType', dummy_na=False)
    else:
        logger.warning("'DayType' column not found in df_opt_period. Cannot create dummy variables for it.")

    df_opt_period = df_opt_period[
        (df_opt_period["date"] >= pd.to_datetime(start_optimize)) & 
        (df_opt_period["date"] <= pd.to_datetime(end_optimize))
    ].reset_index(drop=True)

    if df_opt_period.empty:
        logger.warning(f"No data for lineage {lineage}, case {case_number} in period {start_optimize}-{end_optimize}. Skipping.")
        return pd.DataFrame(columns=FINAL_OPTIMIZE_OUTOUT_COLUMN)

    core_time_range = list(range(8, 18))
    evaluation_data = []
    current_month_processed = -1
    month_params_tuple = None
    tdb_for_pmv_previous_hour = 24.0 # Initialize with a reasonable default

    for conduct_idx in tqdm(range(len(df_opt_period)), desc=f"最適化を実行中 (Case {case_number}, {lineage})", ncols=100):
        current_row_df_for_params = df_opt_period.iloc[conduct_idx:conduct_idx+1].copy()
        
        if current_row_df_for_params["date"].iloc[0].month != current_month_processed:
            month_params_tuple = extract_master_data_values(
                current_row_df_for_params, 0, optimization_params_sheet, case_weights_sheet
            )
            current_month_processed = current_row_df_for_params["date"].iloc[0].month
        
        (
            _, latest_weight, master_pro, master_com, _,
            weight_pro, weight_com, benchmark_temp,
            temp_range_morning_min, temp_range_morning_max,
            temp_range_afternoon_min, temp_range_afternoon_max,
            temp_range_evening_min, temp_range_evening_max,
            overall_temp_range_min, overall_temp_range_max,
            period_type
        ) = month_params_tuple

        hour = current_row_df_for_params["datetime"].iloc[0].hour
        
            # === ADD THIS DEBUG BLOCK ===
        if conduct_idx < 2: # Print for first 2 hours of the optimization period for this case
            logger.info(f"--- DEBUG G(t) Weights --- Case: {case_number}, Lineage: {lineage}, Hour: {hour}")
            logger.info(f"latest_weight columns: {latest_weight.columns.tolist()}")
            logger.info(f"latest_weight values:\n{latest_weight.to_string()}")

            case_weight_cols = latest_weight.columns # These are the column names from your CaseX sheet

            # Be explicit with expected names for clarity in debug
            expected_wellness_core_name = "ウェルネス重み係数_コアタイム"
            expected_energy_core_name = "省エネ重み係数_コアタイム"
            expected_wellness_overtime_name = "ウェルネス重み係数_残業時"
            expected_energy_overtime_name = "省エネ重み係数_残業時"

            # Check if these exact names exist
            if not all(name in case_weight_cols for name in [expected_wellness_core_name, expected_energy_core_name, expected_wellness_overtime_name, expected_energy_overtime_name]):
                logger.error(f"CRITICAL: One or more expected weight column names NOT FOUND in latest_weight.columns: {case_weight_cols}")
                # You might want to raise an error here or use safe defaults if this happens

            # Use direct names if they are reliable, otherwise ensure positional is correct
            # For safety, let's try to use actual names from your Excel images
            g_t_wellness_core_actual = latest_weight[expected_wellness_core_name].iloc[0]
            g_t_energy_core_actual = latest_weight[expected_energy_core_name].iloc[0]
            g_t_wellness_overtime_actual = latest_weight[expected_wellness_overtime_name].iloc[0]
            g_t_energy_overtime_actual = latest_weight[expected_energy_overtime_name].iloc[0]

            current_weight_energy_gt = float(g_t_energy_core_actual if hour in core_time_range else g_t_energy_overtime_actual)
            current_weight_wellness_gt = float(g_t_wellness_core_actual if hour in core_time_range else g_t_wellness_overtime_actual)

            logger.info(f"Calculated G(t) Weights for Hour {hour}: W_Energy_GT={current_weight_energy_gt}, W_Wellness_GT={current_weight_wellness_gt}")
            logger.info(f"Internal Wellness Weights: weight_pro={weight_pro}, weight_com={weight_com}")
            logger.info(f"---------------------------")
        # === END DEBUG BLOCK ===


        active_temp_range = []
        if 8 <= hour <= 10: active_temp_range = [i * 0.5 for i in range(int(temp_range_morning_min * 2), int(temp_range_morning_max * 2) + 1)]
        elif 11 <= hour <= 15: active_temp_range = [i * 0.5 for i in range(int(temp_range_afternoon_min * 2), int(temp_range_afternoon_max * 2) + 1)]
        elif 16 <= hour <= 19: active_temp_range = [i * 0.5 for i in range(int(temp_range_evening_min * 2), int(temp_range_evening_max * 2) + 1)]
        else: active_temp_range = [i * 0.5 for i in range(int(overall_temp_range_min * 2), int(overall_temp_range_max * 2) + 1)]
        if not active_temp_range: active_temp_range = [24.0] # Default if no range is defined

        max_g = -np.inf
        best_temp_for_hour = benchmark_temp 
        best_energy, best_comfort, best_prod = np.nan, np.nan, np.nan
        best_wellness, best_wellness_only_comfort = np.nan, np.nan
        best_pmv, best_rh = np.nan, np.nan
        considered_prod_flag = 0
        
        case_weight_cols = latest_weight.columns
        g_t_wellness_core_col = case_weight_cols[2] 
        g_t_energy_core_col = case_weight_cols[3]
        g_t_wellness_overtime_col = case_weight_cols[4]
        g_t_energy_overtime_col = case_weight_cols[5]

        weight_energy_gt = float(latest_weight[g_t_energy_core_col].iloc[0] if hour in core_time_range else latest_weight[g_t_energy_overtime_col].iloc[0])
        weight_wellness_gt = float(latest_weight[g_t_wellness_core_col].iloc[0] if hour in core_time_range else latest_weight[g_t_wellness_overtime_col].iloc[0])
        
        bm_energy, bm_comfort, bm_prod, bm_wellness, bm_wellness_only_comfort, bm_g_val = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        temp_iter_df = current_row_df_for_params.copy()

        for temp_setpoint_iter in active_temp_range:
            temp_iter_df[current_setpoint_column_name] = temp_setpoint_iter
            
            energy = run_prediction_logic(temp_iter_df, input_features_columns, 0, model)
            prod = calculate_productivity(temp_iter_df, current_setpoint_column_name, master_pro)
            comfort, pmv, rh = calculate_comfort(temp_iter_df, current_setpoint_column_name, master_com, tdb_for_pmv_previous_hour)

            norm_energy = normalize_data(energy, df_for_normalize["消費電力量"])
            norm_comfort = normalize_data(comfort, df_for_normalize["快適性指標"])
            norm_prod = normalize_data(prod, df_for_normalize["知的生産性"])
            
            current_considered_prod_flag = 0
            w_pro_f, w_com_f = float(weight_pro), float(weight_com)

            if pd.isna(prod) or pd.isna(norm_prod) or not (w_pro_f > 0):
                wellness = norm_comfort * w_com_f if pd.notna(norm_comfort) else np.nan
                wellness_only_comfort = wellness
            else:
                current_considered_prod_flag = 1
                term1 = norm_prod * w_pro_f if pd.notna(norm_prod) else 0.0
                term2 = norm_comfort * w_com_f if pd.notna(norm_comfort) else 0.0
                wellness = term1 + term2
                if pd.isna(term1) and pd.isna(term2): wellness = np.nan
                wellness_only_comfort = norm_comfort * w_com_f if pd.notna(norm_comfort) else np.nan
            
            G_t_iter = np.nan
            if pd.notna(norm_energy) and pd.notna(wellness):
                 G_t_iter = (weight_energy_gt * (1 - norm_energy)) + (weight_wellness_gt * wellness)

            if pd.notna(G_t_iter) and G_t_iter > max_g:
                max_g = G_t_iter; best_temp_for_hour = temp_setpoint_iter
                best_energy, best_comfort, best_prod = energy, comfort, prod
                best_wellness, best_wellness_only_comfort = wellness, wellness_only_comfort
                best_pmv, best_rh = pmv, rh; considered_prod_flag = current_considered_prod_flag

            if temp_setpoint_iter == benchmark_temp:
                bm_energy, bm_comfort, bm_prod = energy, comfort, prod
                bm_wellness, bm_wellness_only_comfort = wellness, wellness_only_comfort
                bm_g_val = G_t_iter
        
        if max_g == -np.inf: 
            best_temp_for_hour = benchmark_temp 
            temp_iter_df[current_setpoint_column_name] = best_temp_for_hour
            best_energy = run_prediction_logic(temp_iter_df, input_features_columns, 0, model)
            best_prod = calculate_productivity(temp_iter_df, current_setpoint_column_name, master_pro)
            best_comfort, best_pmv, best_rh = calculate_comfort(temp_iter_df, current_setpoint_column_name, master_com, tdb_for_pmv_previous_hour)
            norm_e,norm_c,norm_p = normalize_data(best_energy,df_for_normalize["消費電力量"]),normalize_data(best_comfort,df_for_normalize["快適性指標"]),normalize_data(best_prod,df_for_normalize["知的生産性"])
            w_p_f, w_c_f = float(weight_pro), float(weight_com)
            if pd.isna(best_prod) or pd.isna(norm_p) or not (w_p_f > 0):
                best_wellness = norm_c * w_c_f if pd.notna(norm_c) else np.nan; best_wellness_only_comfort = best_wellness; considered_prod_flag = 0
            else:
                t1,t2 = (norm_p*w_p_f if pd.notna(norm_p) else 0.0),(norm_c*w_c_f if pd.notna(norm_c) else 0.0)
                best_wellness = t1 + t2; best_wellness_only_comfort = norm_c * w_c_f if pd.notna(norm_c) else np.nan; considered_prod_flag = 1
                if pd.isna(t1) and pd.isna(t2): best_wellness = np.nan
            if pd.notna(norm_e) and pd.notna(best_wellness): max_g = (weight_energy_gt * (1 - norm_e)) + (weight_wellness_gt * best_wellness)
            else: max_g = np.nan

        tdb_for_pmv_previous_hour = best_temp_for_hour # Update for next iteration's PMV

        # Ensure necessary columns for System_ON_OFF and Operation_Mode exist
        on_off_col = f"System_ON_OFF_{lineage[:2]}_{lineage[2:]}"
        op_mode_col = f"Operation_Mode_{lineage[:2]}_{lineage[2:]}"
        on_off_val = current_row_df_for_params[on_off_col].iloc[0] if on_off_col in current_row_df_for_params else np.nan
        op_mode_val = current_row_df_for_params[op_mode_col].iloc[0] if op_mode_col in current_row_df_for_params else np.nan


        eval_entry = {
            "Time": conduct_idx, "datetime": current_row_df_for_params["datetime"].iloc[0],
            "設定温度 [℃]": best_temp_for_hour, "設定PMV [-]": best_pmv,
            "外気温度 [℃]": current_row_df_for_params["外気温度予測値_℃"].iloc[0],
            "予測消費電力量 [kWh]": best_energy, "予測快適性指標(100-PPD) [-]": best_comfort,
            "予測知的生産性 [-]": best_prod, "予測ウェルネス指数 [-]": best_wellness,
            "知的生産性を除いたウェルネス指標値（快適性指標値）": best_wellness_only_comfort,
            "PMV計算時温度": evaluation_data[conduct_idx - 1]["設定温度 [℃]"] if conduct_idx > 0 else best_temp_for_hour, 
            "PMV計算時湿度": best_rh,
            "PMV計算時気流": v_relative(v=master_com[master_com.columns[3]].iloc[0], met=1.0), # air_velocity
            "PMV計算時着衣量": master_com[master_com.columns[2]].iloc[0], # clo
            "PMV計算時活動量": master_com[master_com.columns[1]].iloc[0], # met
            "On/Off": on_off_val,
            "冷暖房モード": op_mode_val,
            "空調期間": period_type, "知的生産性考慮可否": considered_prod_flag,
            "設定温度_ベンチマーク[℃]": benchmark_temp, "予測消費電力量_ベンチマーク [kWh]": bm_energy,
            "予測ウェルネス指数_ベンチマーク [-]": bm_wellness, "予測知的生産性_ベンチマーク [-]": bm_prod,
            "予測快適性指標(100-PPD)_ベンチマーク [-]": bm_comfort,
            "知的生産性を除いたウェルネス指標値（快適性指標値）_ベンチマーク": bm_wellness_only_comfort,
            "評価関数G(t)": max_g, "評価関数G(t)_ベンチマーク": bm_g_val,
        }
        evaluation_data.append(eval_entry)

    output_df = pd.DataFrame(evaluation_data)
    final_output_df = pd.DataFrame() # Create an empty DF first
    for col in FINAL_OPTIMIZE_OUTOUT_COLUMN: # Iterate through the desired columns
        if col in output_df.columns:
            final_output_df[col] = output_df[col]
        else:
            final_output_df[col] = np.nan # Add if missing
    
    if not df_opt_period.empty:
        start_date_str = pd.to_datetime(df_opt_period["date"].iloc[0]).strftime("%Y%m%d")
        end_date_str = pd.to_datetime(df_opt_period["date"].iloc[-1]).strftime("%Y%m%d")
    else: # Fallback if df_opt_period was empty
        start_date_str = pd.to_datetime(start_optimize).strftime("%Y%m%d")
        end_date_str = pd.to_datetime(end_optimize).strftime("%Y%m%d")

    output_filename_new = f"{lineage_for_filename}_{start_date_str}_{end_date_str}_case{case_number}.csv"
    output_path_key = "output_folder_path"
    output_base_path = get_path_from_config_for_outside_base_folder(output_path_key)
    if not os.path.exists(output_base_path): os.makedirs(output_base_path)
    full_output_path = os.path.join(output_base_path, output_filename_new)
    
    logger.info(f"出力結果のファイルパス: {full_output_path}")
    final_output_df.to_csv(full_output_path, encoding="cp932", index=False)

    optimize_time_taken = int(time.time() - start_time_opt_logic)
    logger.info(f"最適化完了 (Case {case_number}, {lineage})。時間={optimize_time_taken // 60}分{optimize_time_taken % 60}秒")
    return final_output_df