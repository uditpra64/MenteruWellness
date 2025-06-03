import numpy as np
import pandas as pd
from data_module.config.utils import get_path_from_config
from prediction_module.electricity_module.prediction_logic import run_prediction_logic
from prediction_module.utilities.models import XGBoostModel
from prediction_module.wellness_module.comfort_logic import calculate_comfort
from prediction_module.wellness_module.productivity_logic import calculate_productivity
from tqdm import tqdm


def create_df_for_normalize(
    df: pd.DataFrame,
    model: XGBoostModel,
    lineage: str,
    start_study_date: str = None,
    end_study_date: str = None,
    input_features_columns: list[str] = None,
    temperature_setpoints_columns: list[str] = None,
) -> pd.DataFrame:
    """
    予測に用いる変数を系統別にまとめたファイルを作成する関数

    Args:
        df (pd.DataFrame): データフレーム
        model (XGBoostModel): モデル
        lineage (str): 系統
        start_study_date (str): 学習期間の開始日
        end_study_date (str): 学習期間の終了日
    """
    # object型をdatatime型に変換
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["date"] = pd.to_datetime(df["date"])
    # 曜日データをダミー変数化
    df = pd.get_dummies(df, dtype=int)
    df = df[
        (df["date"] >= start_study_date) & (df["date"] <= end_study_date)
    ]  # テスト期間の抽出
    df = df.reset_index(drop=True)

    data_for_normalize = []

    master_pro_path = get_path_from_config("master_data_path")
    # Read with header=1 to skip merged cells
    master_data = pd.read_excel(master_pro_path, sheet_name=None, header=1)
    
    lineage = lineage.replace("_", "")
    used_column_list = input_features_columns
    other_column = temperature_setpoints_columns

    # データフレームの各行をループ
    for time in tqdm(
        range(len(df)), desc=f"df_for_normalize_{lineage}を作成中", ncols=100
    ):
        extracted_master_data = extract_master_data(df, time, master_data["最適化"])
        
        # Get master_pro and master_com from the extracted data using positional access
        master_pro = extracted_master_data.iloc[0:1, 2:11]
        master_com = extracted_master_data.iloc[0:1, 17:21]
        
        # Check that df remains a DataFrame at all times
        assert isinstance(
            df, pd.DataFrame
        ), f"df is not a DataFrame at time index {time}"

        energy_consumption = run_prediction_logic(df, used_column_list, time, model)

        productivity_value = calculate_productivity(df, other_column, master_pro, time)

        comfort_value, _, _ = calculate_comfort(df, other_column, master_com, time)
        comfort_value_exp = np.exp(0.05 * comfort_value)

        data_for_normalize.append(
            {
                "datetime": df.loc[time, "datetime"],
                "消費電力量": energy_consumption,
                "快適性指標": comfort_value,
                "快適性指標_exp": comfort_value_exp,
                "知的生産性": productivity_value,
            }
        )

    # データを保存
    df_for_normalize = pd.DataFrame(data_for_normalize)
    df_for_normalize_out_path = (
        get_path_from_config("df_for_normalize_path")
        + f"/df_for_normalize_{lineage}.csv"
    )
    df_for_normalize.to_csv(
        df_for_normalize_out_path,
        index=False,
        encoding="cp932",
    )
    print(f"df_for_normalize out path: {df_for_normalize_out_path}")

    return df_for_normalize


def normalize_data(data: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
    各指標を正規化する関数

    Args:
        data (pd.DataFrame): データフレーム
        column_name (str): 列名
    """
    return (data - column_name.min()) / (column_name.max() - column_name.min())


def extract_master_data(
    df: pd.DataFrame, time: int, master_data: pd.DataFrame
) -> pd.DataFrame:
    """
    月別に必要なマスタデータを抽出する関数

    Args:
        df (pd.DataFrame): データフレーム
        time (int): データフレームの各行のインデックス（タイムステップ）
        master_data (pd.DataFrame): マスタデータフレーム
    """
    # Get month from the data
    current_month = df["date"][time].date().month
    
    # With header=1, the first column is "Unnamed: 0" which contains the month
    month_col = master_data.columns[0]
    
    # Filter for the current month
    month_data = master_data[master_data[month_col] == current_month]
    
    if month_data.empty:
        print(f"Warning: No data found for month {current_month}, using first row as default")
        month_data = master_data.iloc[0:1]
    
    return month_data.reset_index(drop=True)