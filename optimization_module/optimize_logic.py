import time
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
from prediction_module.utilities.models import XGBoostModel
from prediction_module.wellness_module.comfort_logic import calculate_comfort
from prediction_module.wellness_module.productivity_logic import calculate_productivity
from pythermalcomfort.utilities import v_relative
from tqdm import tqdm


# 最適化を実行する関数
def run_optimization_logic(
    df: pd.DataFrame,
    input_features_columns: list[str],
    temperature_setpoints_columns: list[str],
    lineage: str,
    start_study_date: str,
    end_study_date: str,
    start_optimize: str,
    end_optimize: str,
    model: XGBoostModel,
    master_data: dict,
    train_memory_flag: bool = True,
    case: str = None,  # Added case parameter
) -> pd.DataFrame:
    """最適化ロジックを実行する関数

    ステップ
    1. df_for_normalizeのデータを作成
    2. 各時間ステップごとに最適な設定温度を決定
    3. 評価関数G(t)を計算
    4. 評価関数G(t)が最大となる設定温度を探索
    5. 結果をParameterOutputファイルに保存

    Args:
        lineage (str): 選択された系統
        df (pd.DataFrame): 学習データフレーム
        start_study_date (str): 学習期間の開始日
        end_study_date (str): 学習期間の終了日
        start_optimize (str): 最適化期間の開始日
        end_optimize (str): 最適化期間の終了日
        model (XGBoostModel): 学習モデル
        master_data (dict): マスターデータフレーム　（xlsxファイル）
        case (str, optional): 最適化ケース (Case1~Case5)
    """

    start = time.time()
    lineage = lineage.replace("_", "")
    if not train_memory_flag:
        # df_for_normalizeのデータフレームを作成
        df_for_normalize = create_df_for_normalize(
            df,
            model,
            lineage,
            start_study_date,
            end_study_date,
            input_features_columns,
            temperature_setpoints_columns,
        )
    else:
        print(
            "train_memory_flagがTrueのため、csvファイルからdf_for_normalizeを読み込みます"
        )
        df_for_normalize_path = get_path_from_config("df_for_normalize_path")
        df_for_normalize = pd.read_csv(
            df_for_normalize_path + f"/df_for_normalize_{lineage}.csv", encoding="cp932"
        )

    # object型をdatatime型に変換
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["date"] = pd.to_datetime(df["date"])

    df = pd.get_dummies(df, dtype=int)  # 曜日データをダミー変数化

    df = df[
        (df["date"] >= start_optimize) & (df["date"] <= end_optimize)
    ]  # テスト期間の抽出
    df = df.reset_index(drop=True)

    # コアタイムのレンジを定義
    core_time_range = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    temperature_range_min, temperature_range_max = None, None
    # 各時間ステップごとに最適な設定温度を決定
    evaluation_data = []
    for conduct_time in tqdm(range(len(df)), desc="最適化を実行中", ncols=100):
        max_g = None
        best_temp = None

        # 最適化の初回実行時(conduct_time=0)にマスターデータから必要なパラメータを取得
        # 温度範囲、重み係数、ベンチマーク温度などの最適化に必要な値を設定
        if conduct_time == 0:
            (
                month,
                latest_weight,
                master_pro,
                master_com,
                temperature_range,
                weight_pro,
                weight_com,
                benchmark_temp,
                temp_range_morning_min,
                temp_range_morning_max,
                temp_range_afternoon_min,
                temp_range_afternoon_max,
                temp_range_evening_min,
                temp_range_evening_max,
                temperature_range_min,
                temperature_range_max,
                extracted_master_data,
            ) = extract_master_data_values(df, conduct_time, master_data, case)
        else:
            # 月が変わった場合(conduct_time>0)、その月に対応するマスターデータのパラメータを再取得
            new_month = df["date"][conduct_time].date().month
            if month != new_month:
                (
                    month,
                    latest_weight,
                    master_pro,
                    master_com,
                    temperature_range,
                    weight_pro,
                    weight_com,
                    benchmark_temp,
                    temp_range_morning_min,
                    temp_range_morning_max,
                    temp_range_afternoon_min,
                    temp_range_afternoon_max,
                    temp_range_evening_min,
                    temp_range_evening_max,
                    temperature_range_min,
                    temperature_range_max,
                    extracted_master_data,
                ) = extract_master_data_values(df, conduct_time, master_data, case)

        hour = int(
            pd.to_datetime(df.loc[conduct_time, "datetime"]).time().strftime("%H")
        )
        if 8 <= hour <= 10:
            temperature_range = [
                i * 0.5
                for i in range(
                    int(temp_range_morning_min * 2), int(temp_range_morning_max * 2) + 1
                )
            ]
        elif 13 <= hour <= 15:
            temperature_range = [
                i * 0.5
                for i in range(
                    int(temp_range_afternoon_min * 2),
                    int(temp_range_afternoon_max * 2) + 1,
                )
            ]
        elif 16 <= hour <= 19:
            temperature_range = [
                i * 0.5
                for i in range(
                    int(temp_range_evening_min * 2), int(temp_range_evening_max * 2) + 1
                )
            ]
        else:
            temperature_range = [
                i * 0.5
                for i in range(
                    int(temperature_range_min * 2), int(temperature_range_max * 2) + 1
                )
            ]

        actual_comfort, _, _ = calculate_comfort(
            df, temperature_setpoints_columns, master_com, conduct_time
        )
        actual_productivity = calculate_productivity(
            df, temperature_setpoints_columns, master_pro, conduct_time
        )
        acutual_normalized_comfort = normalize_data(
            actual_comfort, df_for_normalize["快適性指標"]
        )
        acutual_normalized_comfort = normalize_data(
            actual_comfort, df_for_normalize["快適性指標"]
        )
        actual_normalized_productivity = normalize_data(
            actual_productivity, df_for_normalize["知的生産性"]
        )
        if np.isnan(actual_productivity):
            actual_wellness = acutual_normalized_comfort
        else:
            actual_wellness = (
                acutual_normalized_comfort + actual_normalized_productivity
            ) / 2

        # コアタイムの時の重みを取得
        if (
            int(pd.to_datetime(df.loc[conduct_time, "datetime"]).time().strftime("%H"))
            in core_time_range
        ):
            weight_energy = latest_weight["省エネ重み係数_コアタイム"].iloc[0]
            weight_wellness = latest_weight["ウェルネス重み係数_コアタイム"].iloc[0]
        else:
            weight_energy = latest_weight["省エネ重み係数_残業時"].iloc[0]
            weight_wellness = latest_weight["ウェルネス重み係数_残業時"].iloc[0]

        # Initialize benchmark variables with default values
        benchmark_energy_consumption = 0
        benchmark_wellness_value = 0
        benchmark_comfort_value = 0
        benchmark_productivity_value = 0
        benchmark_wellness_value_only_comfort = 0
        benchmark_g = 0
        
        # Set a flag to check if benchmark was found
        benchmark_found = False

        for temp in temperature_range:
            df[temperature_setpoints_columns] = temp
            energy_consumption = run_prediction_logic(
                df, input_features_columns, conduct_time, model
            )  # 消費電力量の予測
            productivity_value = calculate_productivity(
                df, temperature_setpoints_columns, master_pro, conduct_time
            )  # 知的生産性指標の予測

            comfort_value, _, indoor_relative_humidity = calculate_comfort(
                df, temperature_setpoints_columns, master_com, conduct_time
            )  # 快適性指標の予測

            # 上記で作成したデータフレームを利用して，標準化を行う
            normalized_energy_consumption = normalize_data(
                energy_consumption, df_for_normalize["消費電力量"]
            )
            normalized_comfort_value = normalize_data(
                comfort_value, df_for_normalize["快適性指標"]
            )
            normalized_productivity_value = normalize_data(
                productivity_value, df_for_normalize["知的生産性"]
            )

            # ウェルネス指標は快適性指標と知的生産性を1:1で重みづけ
            # productivtyがnanの場合は，ウェルネス指標は快適性指標と同じ値になる
            if np.isnan(productivity_value):
                considered_productivity = 0
                wellness_value = normalized_comfort_value
            else:

                considered_productivity = 1
                wellness_value = (
                    normalized_productivity_value * weight_pro
                    + normalized_comfort_value * weight_com
                )

            wellness_value_only_comfort = normalized_comfort_value * weight_com
            # 評価関数 G(t) を計算
            if weight_wellness > 0:
                G_t = (
                    weight_energy * (1 - normalized_energy_consumption)
                    + weight_wellness * wellness_value
                )
            else:
                G_t = (weight_energy - 0.00000000000001) * (
                    1 - normalized_energy_consumption
                ) + (weight_wellness + 0.00000000000001) * wellness_value

            # G(t)が最大となる設定温度を探索
            if max_g is None or G_t > max_g:
                max_g = G_t
                best_temp = temp
                best_energy_consumption = energy_consumption
                best_normalized_energy_consumption = normalized_energy_consumption
                best_comfort_value = comfort_value
                best_productivity_value = productivity_value
                best_wellness_value = wellness_value
                best_wellness_value_only_comfort = wellness_value_only_comfort
                pmv_temp = (
                    best_temp
                    if conduct_time == 0
                    else evaluation_data[conduct_time - 1]["設定温度 [℃]"]
                )

            # ベンチマークの設定温度での消費電力量を計算
            if temp == benchmark_temp:
                benchmark_energy_consumption = energy_consumption
                benchmark_wellness_value = wellness_value
                benchmark_comfort_value = comfort_value
                benchmark_productivity_value = productivity_value
                benchmark_wellness_value_only_comfort = wellness_value_only_comfort
                benchmark_g = G_t
                benchmark_found = True

        # After the loop, if benchmark wasn't found, attempt to calculate it
        if not benchmark_found:
            print(f"Warning: Benchmark temp {benchmark_temp} not found in temperature range {temperature_range[0]} to {temperature_range[-1]}. Using closest value.")
            
            # Find the closest temperature in the range to the benchmark
            closest_temp = min(temperature_range, key=lambda x: abs(x - benchmark_temp))
            
            # Calculate values for the closest temperature
            df[temperature_setpoints_columns] = closest_temp
            benchmark_energy_consumption = run_prediction_logic(
                df, input_features_columns, conduct_time, model
            )
            benchmark_productivity_value = calculate_productivity(
                df, temperature_setpoints_columns, master_pro, conduct_time
            )
            benchmark_comfort_value, _, _ = calculate_comfort(
                df, temperature_setpoints_columns, master_com, conduct_time
            )
            
            # Normalize values
            normalized_benchmark_energy = normalize_data(
                benchmark_energy_consumption, df_for_normalize["消費電力量"]
            )
            normalized_benchmark_comfort = normalize_data(
                benchmark_comfort_value, df_for_normalize["快適性指標"]
            )
            normalized_benchmark_productivity = normalize_data(
                benchmark_productivity_value, df_for_normalize["知的生産性"]
            )
            
            # Calculate wellness value
            if np.isnan(benchmark_productivity_value):
                benchmark_wellness_value = normalized_benchmark_comfort
            else:
                benchmark_wellness_value = (
                    normalized_benchmark_productivity * weight_pro
                    + normalized_benchmark_comfort * weight_com
                )
            
            benchmark_wellness_value_only_comfort = normalized_benchmark_comfort * weight_com
            
            # Calculate G(t)
            if weight_wellness > 0:
                benchmark_g = (
                    weight_energy * (1 - normalized_benchmark_energy)
                    + weight_wellness * benchmark_wellness_value
                )
            else:
                benchmark_g = (weight_energy - 0.00000000000001) * (
                    1 - normalized_benchmark_energy
                ) + (weight_wellness + 0.00000000000001) * benchmark_wellness_value

        evaluation_data.append(
            {
                "Time": conduct_time,
                "datetime": df.loc[conduct_time, "datetime"],
                "設定温度 [℃]": best_temp,
                "予測消費電力量 [kWh]": best_energy_consumption,
                "予測快適性指標(100-PPD) [-]": best_comfort_value,
                "予測知的生産性 [-]": best_productivity_value,
                "予測ウェルネス指数 [-]": best_wellness_value,
                "予測消費電力量_正規化": best_normalized_energy_consumption,
                "On/Off": df.loc[
                    conduct_time,
                    f"System_ON_OFF_{lineage[:2]}_{lineage[2:]}",
                ],
                "冷暖房モード": df.loc[
                    conduct_time,
                    f"Operation_Mode_{lineage[:2]}_{lineage[2:]}",
                ],
                "設定温度_ベンチマーク[℃]": benchmark_temp,
                "予測消費電力量_ベンチマーク [kWh]": benchmark_energy_consumption,
                "予測ウェルネス指数_ベンチマーク [-]": benchmark_wellness_value,
                "外気温度 [℃]": df.loc[conduct_time, "外気温度予測値_℃"],
                "実績値の消費電力量 [-]": df.loc[
                    conduct_time,
                    df.columns[df.columns.str.contains("室内機消費電力量")][0],
                ]
                + df.loc[
                    conduct_time,
                    df.columns[df.columns.str.contains("室外機消費電力量")][0],
                ],
                "実績値のウェルネス指数 [-]": actual_wellness,
                "評価関数G(t)": max_g,
                "評価関数G(t)_ベンチマーク": benchmark_g,
                "PMV計算時温度": pmv_temp,  # best_temp のt - 1ですが, t=0の場合　pmv_temp = best_temp
                "PMV計算時湿度": indoor_relative_humidity,
                "PMV計算時気流": v_relative(v=master_com.iloc[0, 3], met=1),
                "PMV計算時着衣量": master_com.iloc[0, 2],
                "PMV計算時活動量": master_com.iloc[0, 1],
                "空調期間": extracted_master_data.loc[0, "期間区分"] if "期間区分" in extracted_master_data.columns else "unknown",
                "知的生産性考慮可否": considered_productivity,
                # "設定PMV [-]": pmv_value,
                "予測知的生産性_ベンチマーク [-]": benchmark_productivity_value,
                "予測快適性指標(100-PPD)_ベンチマーク [-]": benchmark_comfort_value,
                "知的生産性を除いたウェルネス指標値（快適性指標値）": best_wellness_value_only_comfort,
                "知的生産性を除いたウェルネス指標値（快適性指標値）_ベンチマーク": benchmark_wellness_value_only_comfort,
                f"設定温度_C_{lineage[:2]}執務室_{lineage[2:]}": best_temp,
            }
        )

    # ここでは，calculate_comfort関数を利用して，設定PMV [-]を計算
    eval_df = pd.DataFrame(evaluation_data)
    pmv_values = []
    for t in range(len(eval_df)):
        _, pmv_value, _ = calculate_comfort(
            eval_df, temperature_setpoints_columns, master_com, t
        )
        pmv_values.append(pmv_value)
        evaluation_data[t]["設定PMV [-]"] = pmv_value

    # 結果を保存
    output_df = pd.DataFrame(evaluation_data)
    start_date = df["date"].iloc[0].strftime("%Y%m%d")
    end_date = df["date"].iloc[-1].strftime("%Y%m%d")
    output_path = get_path_from_config_for_outside_base_folder("output_folder_path")
    
    # Include case in the output filename if provided
    if case:
        output_filename = output_path + f"/{lineage}_{case}_{start_date}_to_{end_date}.csv"
    else:
        output_filename = output_path + f"/{lineage}_{start_date}_to_{end_date}.csv"
    
    print(f"出力結果のファイルパス: {output_filename}")

    output_df = output_df[FINAL_OPTIMIZE_OUTOUT_COLUMN]  # 最終的な出力カラムのみを抽出
    output_df.to_csv(output_filename, encoding="cp932", index=False)

    optimize_time = int(time.time() - start)  # 最適化に要した時間を計算
    print(
        f"最適化が完了しました。要した時間={optimize_time // 60} 分 {optimize_time % 60} 秒"
    )


def extract_master_data_values(
    df: pd.DataFrame, conduct_time: int, master_data: dict, case: str = None
) -> Tuple:
    """
    Extracts necessary values from the master data based on the conduct time.
    Now handles multi-row headers properly.

    Args:
        df (pd.DataFrame): The DataFrame containing date information.
        conduct_time (int): The current time index used for extraction.
        master_data (dict): Dictionary containing master data sheets for optimization.
        case (str, optional): The optimization case (Case1-Case5) to use.

    Returns:
        Tuple containing various optimization parameters.
    """
    month = df["date"][conduct_time].date().month
    
    try:
        # Load master data sheets without headers
        if "最適化" in master_data:
            master_data_opt = master_data["最適化"]
            
            # If it's already a DataFrame with no headers, use it directly
            if isinstance(master_data_opt, pd.DataFrame):
                # Try to find the data rows (look for month column with values 1-12)
                data_rows = master_data_opt[master_data_opt.iloc[:, 0].isin(range(1, 13))]
                
                if not data_rows.empty:
                    master_opt_data = data_rows.reset_index(drop=True)
                else:
                    # Fallback - assume second row is header, data starts from third row
                    master_opt_data = master_data_opt.iloc[2:].reset_index(drop=True)
                    
                # Find rows containing current month
                month_rows = master_opt_data[master_opt_data.iloc[:, 0] == month]
                
                if month_rows.empty:
                    print(f"Month {month} not found in optimization data, using first row")
                    extracted_master_data = master_opt_data.iloc[0:1].copy()
                else:
                    extracted_master_data = month_rows.iloc[0:1].copy()
                
                # Set column names from the header row
                header_row = master_data_opt.iloc[1] if len(master_data_opt) > 1 else pd.Series(["月", "期間区分"])
                extracted_master_data.columns = header_row
            else:
                # If not a DataFrame, create an empty one with appropriate columns
                print("Error: 最適化 sheet is not a DataFrame")
                extracted_master_data = pd.DataFrame({
                    "月": [month],
                    "期間区分": ["unknown"]
                })
        else:
            print("Error: 最適化 sheet not found")
            extracted_master_data = pd.DataFrame({
                "月": [month],
                "期間区分": ["unknown"]
            })
            
        # Default weight coefficients
        latest_weight = pd.DataFrame({
            "ウェルネス重み係数_コアタイム": [0.9],
            "省エネ重み係数_コアタイム": [0.1],
            "ウェルネス重み係数_残業時": [0.9],
            "省エネ重み係数_残業時": [0.1]
        })
    
        # If a case is specified, get weights from the case-specific sheet
        if case and f"重み係数_{case}" in master_data:
            try:
                case_sheet = master_data[f"重み係数_{case}"]
                
                # Find data rows (look for month column with values 1-12)
                data_rows = case_sheet[case_sheet.iloc[:, 0].isin(range(1, 13))]
                
                if not data_rows.empty:
                    case_data = data_rows.reset_index(drop=True)
                else:
                    # Fallback - assume second row is header, data starts from third row
                    case_data = case_sheet.iloc[2:].reset_index(drop=True)
                
                # Find rows containing current month
                month_rows = case_data[case_data.iloc[:, 0] == month]
                
                if not month_rows.empty:
                    # Extract weight values from columns 2-5
                    latest_weight = pd.DataFrame({
                        "ウェルネス重み係数_コアタイム": [month_rows.iloc[0, 2]],
                        "省エネ重み係数_コアタイム": [month_rows.iloc[0, 3]],
                        "ウェルネス重み係数_残業時": [month_rows.iloc[0, 4]],
                        "省エネ重み係数_残業時": [month_rows.iloc[0, 5]]
                    })
                    print(f"Using weights from {case} for month {month}")
                else:
                    print(f"No data found for month {month} in {case} sheet, using default weights")
            except Exception as e:
                print(f"Error extracting weights for {case}: {e}. Using default weights.")
                    
    except Exception as e:
        print(f"Error in extract_master_data_values: {e}")
        # Create fallback data
        extracted_master_data = pd.DataFrame({
            "月": [month],
            "期間区分": ["unknown"]
        })
        latest_weight = pd.DataFrame({
            "ウェルネス重み係数_コアタイム": [0.9],
            "省エネ重み係数_コアタイム": [0.1],
            "ウェルネス重み係数_残業時": [0.9],
            "省エネ重み係数_残業時": [0.1]
        })
        
    # Create DataFrames for productivity and comfort parameters using positional values
    # from the extracted master data
    
    # Master Pro - Intellectual productivity parameters
    # Since there could be missing columns in the extracted data, create with default values
    try:
        master_pro = pd.DataFrame({
            "知的生産性_朝_A": [extracted_master_data.iloc[0, 2] if extracted_master_data.shape[1] > 2 else 0],
            "知的生産性_朝_B": [extracted_master_data.iloc[0, 3] if extracted_master_data.shape[1] > 3 else 0],
            "知的生産性_朝_C": [extracted_master_data.iloc[0, 4] if extracted_master_data.shape[1] > 4 else 0],
            "知的生産性_昼_A": [extracted_master_data.iloc[0, 5] if extracted_master_data.shape[1] > 5 else 0],
            "知的生産性_昼_B": [extracted_master_data.iloc[0, 6] if extracted_master_data.shape[1] > 6 else 0],
            "知的生産性_昼_C": [extracted_master_data.iloc[0, 7] if extracted_master_data.shape[1] > 7 else 0],
            "知的生産性_夕_A": [extracted_master_data.iloc[0, 8] if extracted_master_data.shape[1] > 8 else 0],
            "知的生産性_夕_B": [extracted_master_data.iloc[0, 9] if extracted_master_data.shape[1] > 9 else 0],
            "知的生産性_夕_C": [extracted_master_data.iloc[0, 10] if extracted_master_data.shape[1] > 10 else 0],
        })
    except Exception as e:
        print(f"Error creating master_pro: {e}")
        master_pro = pd.DataFrame({
            "知的生産性_朝_A": [0], "知的生産性_朝_B": [0], "知的生産性_朝_C": [0],
            "知的生産性_昼_A": [0], "知的生産性_昼_B": [0], "知的生産性_昼_C": [0],
            "知的生産性_夕_A": [0], "知的生産性_夕_B": [0], "知的生産性_夕_C": [0],
        })
    
    # Master Com - Comfort parameters
    try:
        master_com = pd.DataFrame({
            "室内相対湿度 [%]": [extracted_master_data.iloc[0, 17] if extracted_master_data.shape[1] > 17 else 50],
            "代謝量 [met]": [extracted_master_data.iloc[0, 18] if extracted_master_data.shape[1] > 18 else 1.1],
            "着衣量 [clo]": [extracted_master_data.iloc[0, 19] if extracted_master_data.shape[1] > 19 else 0.8],
            "気流速度 [m/s]": [extracted_master_data.iloc[0, 20] if extracted_master_data.shape[1] > 20 else 0.1],
        })
    except Exception as e:
        print(f"Error creating master_com: {e}")
        master_com = pd.DataFrame({
            "室内相対湿度 [%]": [50],
            "代謝量 [met]": [1.1],
            "着衣量 [clo]": [0.8],
            "気流速度 [m/s]": [0.1],
        })
    
    # Extract other parameters
    try:
        weight_pro = float(extracted_master_data.iloc[0, 21] if extracted_master_data.shape[1] > 21 else 0.5)
        weight_com = float(extracted_master_data.iloc[0, 22] if extracted_master_data.shape[1] > 22 else 0.5)
        temperature_range_max = float(extracted_master_data.iloc[0, 23] if extracted_master_data.shape[1] > 23 else 28.0)
        temperature_range_min = float(extracted_master_data.iloc[0, 24] if extracted_master_data.shape[1] > 24 else 20.0)
        benchmark_temp = float(extracted_master_data.iloc[0, 25] if extracted_master_data.shape[1] > 25 else 24.0)
    except Exception as e:
        print(f"Error extracting parameters: {e}")
        weight_pro = 0.5
        weight_com = 0.5
        temperature_range_max = 28.0
        temperature_range_min = 20.0
        benchmark_temp = 24.0
    
    # Temperature range values for different times of day
    try:
        temp_range_morning_min = float(extracted_master_data.iloc[0, 11] if extracted_master_data.shape[1] > 11 else 20.0)
        temp_range_morning_max = float(extracted_master_data.iloc[0, 12] if extracted_master_data.shape[1] > 12 else 28.0)
        temp_range_afternoon_min = float(extracted_master_data.iloc[0, 13] if extracted_master_data.shape[1] > 13 else 20.0)
        temp_range_afternoon_max = float(extracted_master_data.iloc[0, 14] if extracted_master_data.shape[1] > 14 else 28.0)
        temp_range_evening_min = float(extracted_master_data.iloc[0, 15] if extracted_master_data.shape[1] > 15 else 20.0)
        temp_range_evening_max = float(extracted_master_data.iloc[0, 16] if extracted_master_data.shape[1] > 16 else 28.0)
    except Exception as e:
        print(f"Error extracting temperature ranges: {e}")
        temp_range_morning_min = temperature_range_min
        temp_range_morning_max = temperature_range_max
        temp_range_afternoon_min = temperature_range_min
        temp_range_afternoon_max = temperature_range_max
        temp_range_evening_min = temperature_range_min
        temp_range_evening_max = temperature_range_max
    
    # Create temperature range
    temperature_range = [
        i * 0.5
        for i in range(
            int(temperature_range_min * 2), int(temperature_range_max * 2) + 1
        )
    ]

    return (
        month,
        latest_weight,
        master_pro,
        master_com,
        temperature_range,
        weight_pro,
        weight_com,
        benchmark_temp,
        temp_range_morning_min,
        temp_range_morning_max,
        temp_range_afternoon_min,
        temp_range_afternoon_max,
        temp_range_evening_min,
        temp_range_evening_max,
        temperature_range_min,
        temperature_range_max,
        extracted_master_data,
    )