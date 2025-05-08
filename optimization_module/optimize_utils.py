import numpy as np
import pandas as pd
from data_module.config.utils import get_path_from_config
from prediction_module.electricity_module.prediction_logic import run_prediction_logic
from prediction_module.utilities.models import XGBoostModel
from prediction_module.wellness_module.comfort_logic import calculate_comfort
from tqdm import tqdm


def create_df_for_normalize(
    df: pd.DataFrame,
    model: XGBoostModel,
    lineage_key: str,
    start_study_date: str,
    end_study_date: str,
    input_features_columns: list[str],
    temperature_setpoints_columns: list[str],
    master_data_full: pd.DataFrame,
) -> pd.DataFrame:
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["date"]     = pd.to_datetime(df["date"])
    df = pd.get_dummies(df, dtype=int)
    df = df[(df["date"] >= start_study_date) & (df["date"] <= end_study_date)].reset_index(drop=True)

    records = []
    for i in tqdm(range(len(df)), desc=f"df_for_normalize_{lineage_key}", ncols=100):
        combined = extract_master_data(df, i, master_data_full)
        master_pro = combined.iloc[0:1, 6:15]
        master_com = combined.iloc[0:1, 21:25]

        e = run_prediction_logic(df, input_features_columns, i, model)
        c, _, _ = calculate_comfort(df, temperature_setpoints_columns, master_com, i)
        p = calculate_productivity(df, temperature_setpoints_columns, master_pro, i)

        records.append({
            "datetime":      df.loc[i, "datetime"],
            "消費電力量":     e,
            "快適性指標":     c,
            "快適性指標_exp": np.exp(0.05 * c),
            "知的生産性":     p,
        })

    out_df = pd.DataFrame(records)
    out_path = get_path_from_config("df_for_normalize_path")
    filename = f"{out_path}/df_for_normalize_{lineage_key}.csv"
    out_df.to_csv(filename, index=False, encoding="cp932")
    print(f"df_for_normalize out: {filename}")
    return out_df


def normalize_data(data, column_series):
    return (data - column_series.min()) / (column_series.max() - column_series.min())


def extract_master_data(df, idx, master_data_full):
    header = master_data_full.iloc[0:1]
    month_val = df.loc[idx, "date"].month
    row = master_data_full[master_data_full["月"] == month_val]
    combined = pd.concat([header, row], ignore_index=True)

    cols = combined.columns
    combined.at[0, "月"] = cols[0]
    combined.at[0, "期間区分"] = cols[1]
    if "ベンチマークケース設定温度" in cols:
        combined.at[0, "ベンチマークケース設定温度"] = cols[-1]

    combined.columns = combined.iloc[0]
    return combined.drop(0).reset_index(drop=True)
