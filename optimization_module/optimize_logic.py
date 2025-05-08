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
    master_data: pd.DataFrame,
    train_memory_flag: bool = True,
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
        master_data (pd.DataFrame): マスターデータフレーム　（xlsxファイル）
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
            ) = extract_master_data_values(df, conduct_time, master_data)
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
                ) = extract_master_data_values(df, conduct_time, master_data)

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
                "空調期間": extracted_master_data.loc[0, "期間区分"],
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
    output_filename = output_path + f"/{lineage}_{start_date}_to_{end_date}.csv"
    print(f"出力結果のファイルパス: {output_filename}")

    output_df = output_df[FINAL_OPTIMIZE_OUTOUT_COLUMN]  # 最終的な出力カラムのみを抽出
    output_df.to_csv(output_filename, encoding="cp932", index=False)

    optimize_time = int(time.time() - start)  # 最適化に要した時間を計算
    print(
        f"最適化が完了しました。要した時間={optimize_time // 60} 分 {optimize_time % 60} 秒"
    )


def extract_master_data_values(
    df: pd.DataFrame, conduct_time: int, master_data: pd.DataFrame
) -> Tuple:
    """
    Extracts necessary values from the master data based on the conduct time.

    Args:
        df (pd.DataFrame): The DataFrame containing date information.
        conduct_time (int): The current time index used for extraction.
        master_data (pd.DataFrame): DataFrame containing master data for optimization.

    Returns:
        Tuple containing:
            - month (int): Month of the current conduct time.
            - latest_weight (pd.DataFrame): DataFrame with the latest weight coefficients.
            - master_pro (pd.DataFrame): DataFrame with fixed variables for intellectual productivity.
            - master_com (pd.DataFrame): DataFrame with fixed variables for comfort indicators.
            - temperature_range (List[float]): List of temperature ranges for time steps.
            - weight_pro (float): Weight for intellectual productivity.
            - weight_com (float): Weight for comfort indicators.
            - benchmark_temp (float): Benchmark setting temperature.
            - temp_range_morning_min (float): Lower temperature limit for productivity in the morning.
            - temp_range_morning_max (float): Upper temperature limit for productivity in the morning.
            - temp_range_afternoon_min (float): Lower temperature limit for productivity in the afternoon.
            - temp_range_afternoon_max (float): Upper temperature limit for productivity in the afternoon.
            - temp_range_evening_min (float): Lower temperature limit for productivity in the evening.
            - temp_range_evening_max (float): Upper temperature limit for productivity in the evening.
            - temperature_range_min (float): Minimum temperature range.
            - temperature_range_max (float): Maximum temperature range.
            - extracted_master_data (pd.DataFrame): The full extracted master data row, used to access additional fields.
    """
    month = df["date"][conduct_time].date().month
    extracted_master_data = extract_master_data(df, conduct_time, master_data["最適化"])

    latest_weight = extracted_master_data.iloc[0:1, 2:6]
    master_pro = extracted_master_data.iloc[0:1, 6:15]
    master_com = extracted_master_data.iloc[0:1, 21:25]
    temperature_range_max = extracted_master_data.iloc[0, -3]
    temperature_range_min = extracted_master_data.iloc[0, -2]
    temperature_range = [
        i * 0.5
        for i in range(
            int(temperature_range_min * 2), int(temperature_range_max * 2) + 1
        )
    ]
    weight_pro = extracted_master_data.iloc[0, -5]
    weight_com = extracted_master_data.iloc[0, -4]
    benchmark_temp = extracted_master_data.iloc[0, -1]
    temp_range_morning_min = extracted_master_data["知的生産性_下限温度_朝"].iloc[0]
    temp_range_morning_max = extracted_master_data["知的生産性_上限温度_朝"].iloc[0]
    temp_range_afternoon_min = extracted_master_data["知的生産性_下限温度_昼"].iloc[0]
    temp_range_afternoon_max = extracted_master_data["知的生産性_上限温度_昼"].iloc[0]
    temp_range_evening_min = extracted_master_data["知的生産性_下限温度_夕"].iloc[0]
    temp_range_evening_max = extracted_master_data["知的生産性_上限温度_夕"].iloc[0]

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
