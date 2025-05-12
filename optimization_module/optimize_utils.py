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
    # pmv_values = []  # List to store PMV values

    master_pro_path = get_path_from_config("master_data_path")
    try:
        # Read with header=None to handle multi-row headers
        master_data_xl = pd.ExcelFile(master_pro_path)
        if "最適化" in master_data_xl.sheet_names:
            master_data = {"最適化": pd.read_excel(master_data_xl, sheet_name="最適化", header=None)}
            
            # Process other case sheets if they exist
            for case_num in range(1, 6):
                case_sheet = f"重み係数_Case{case_num}"
                if case_sheet in master_data_xl.sheet_names:
                    master_data[case_sheet] = pd.read_excel(master_data_xl, sheet_name=case_sheet, header=None)
        else:
            print("Warning: '最適化' sheet not found in master data")
            master_data = {"最適化": pd.DataFrame()}
    except Exception as e:
        print(f"Error loading master data: {e}")
        master_data = {"最適化": pd.DataFrame()}
    
    lineage = lineage.replace("_", "")
    used_column_list = input_features_columns
    other_column = temperature_setpoints_columns

    # データフレームの各行をループ
    for time in tqdm(
        range(len(df)), desc=f"df_for_normalize_{lineage}を作成中", ncols=100
    ):
        try:
            # Extract master data
            extracted_master_data = extract_master_data(df, time, master_data["最適化"])
            
            # In the extracted_master_data, create a dictionary to map column positions to values
            master_data_dict = {}
            
            # Intellectual productivity coefficients (based on positions from the Excel)
            # Morning
            master_data_dict["知的生産性_朝_A"] = extracted_master_data.iloc[0, 0] if extracted_master_data.shape[1] > 0 else 0
            master_data_dict["知的生産性_朝_B"] = extracted_master_data.iloc[0, 1] if extracted_master_data.shape[1] > 1 else 0
            master_data_dict["知的生産性_朝_C"] = extracted_master_data.iloc[0, 2] if extracted_master_data.shape[1] > 2 else 0
            
            # Afternoon
            master_data_dict["知的生産性_昼_A"] = extracted_master_data.iloc[0, 3] if extracted_master_data.shape[1] > 3 else 0
            master_data_dict["知的生産性_昼_B"] = extracted_master_data.iloc[0, 4] if extracted_master_data.shape[1] > 4 else 0
            master_data_dict["知的生産性_昼_C"] = extracted_master_data.iloc[0, 5] if extracted_master_data.shape[1] > 5 else 0
            
            # Evening
            master_data_dict["知的生産性_夕_A"] = extracted_master_data.iloc[0, 6] if extracted_master_data.shape[1] > 6 else 0
            master_data_dict["知的生産性_夕_B"] = extracted_master_data.iloc[0, 7] if extracted_master_data.shape[1] > 7 else 0
            master_data_dict["知的生産性_夕_C"] = extracted_master_data.iloc[0, 8] if extracted_master_data.shape[1] > 8 else 0
            
            # Temperature ranges
            master_data_dict["知的生産性_下限温度_朝"] = extracted_master_data.iloc[0, 9] if extracted_master_data.shape[1] > 9 else 20
            master_data_dict["知的生産性_上限温度_朝"] = extracted_master_data.iloc[0, 10] if extracted_master_data.shape[1] > 10 else 28
            master_data_dict["知的生産性_下限温度_昼"] = extracted_master_data.iloc[0, 11] if extracted_master_data.shape[1] > 11 else 20
            master_data_dict["知的生産性_上限温度_昼"] = extracted_master_data.iloc[0, 12] if extracted_master_data.shape[1] > 12 else 28
            master_data_dict["知的生産性_下限温度_夕"] = extracted_master_data.iloc[0, 13] if extracted_master_data.shape[1] > 13 else 20
            master_data_dict["知的生産性_上限温度_夕"] = extracted_master_data.iloc[0, 14] if extracted_master_data.shape[1] > 14 else 28
            
            # Comfort parameters
            master_data_dict["室内相対湿度 [%]"] = extracted_master_data.iloc[0, 15] if extracted_master_data.shape[1] > 15 else 50
            master_data_dict["代謝量 [met]"] = extracted_master_data.iloc[0, 16] if extracted_master_data.shape[1] > 16 else 1.1
            master_data_dict["着衣量 [clo]"] = extracted_master_data.iloc[0, 17] if extracted_master_data.shape[1] > 17 else 0.8
            master_data_dict["気流速度 [m/s]"] = extracted_master_data.iloc[0, 18] if extracted_master_data.shape[1] > 18 else 0.1
            
            # Create DataFrame for productivity calculation
            master_pro = pd.DataFrame({
                "知的生産性_朝_A": [master_data_dict["知的生産性_朝_A"]],
                "知的生産性_朝_B": [master_data_dict["知的生産性_朝_B"]],
                "知的生産性_朝_C": [master_data_dict["知的生産性_朝_C"]],
                "知的生産性_昼_A": [master_data_dict["知的生産性_昼_A"]],
                "知的生産性_昼_B": [master_data_dict["知的生産性_昼_B"]],
                "知的生産性_昼_C": [master_data_dict["知的生産性_昼_C"]],
                "知的生産性_夕_A": [master_data_dict["知的生産性_夕_A"]],
                "知的生産性_夕_B": [master_data_dict["知的生産性_夕_B"]],
                "知的生産性_夕_C": [master_data_dict["知的生産性_夕_C"]],
            })
            
            # Create DataFrame for comfort calculation
            master_com = pd.DataFrame({
                "室内相対湿度 [%]": [master_data_dict["室内相対湿度 [%]"]],
                "代謝量 [met]": [master_data_dict["代謝量 [met]"]],
                "着衣量 [clo]": [master_data_dict["着衣量 [clo]"]],
                "気流速度 [m/s]": [master_data_dict["気流速度 [m/s]"]],
            })
            
            # Check that df remains a DataFrame at all times
            assert isinstance(
                df, pd.DataFrame
            ), f"df is not a DataFrame at time index {time}"

            energy_consumption = run_prediction_logic(df, used_column_list, time, model)

            productivity_value = calculate_productivity(df, other_column, master_pro, time)

            comfort_value, _, _ = calculate_comfort(df, other_column, master_com, time)
            comfort_value_exp = np.exp(0.05 * comfort_value) if not np.isnan(comfort_value) else np.nan

            data_for_normalize.append(
                {
                    "datetime": df.loc[time, "datetime"],
                    "消費電力量": energy_consumption if energy_consumption is not None else 0,
                    "快適性指標": comfort_value if not np.isnan(comfort_value) else 50,
                    "快適性指標_exp": comfort_value_exp if not np.isnan(comfort_value_exp) else np.exp(0.05 * 50),
                    "知的生産性": productivity_value if not np.isnan(productivity_value) else 0,
                }
            )
        except Exception as e:
            print(f"Error processing time {time}: {e}")
            # Add a default entry
            data_for_normalize.append(
                {
                    "datetime": df.loc[time, "datetime"] if time < len(df) else pd.Timestamp.now(),
                    "消費電力量": 0,
                    "快適性指標": 50,
                    "快適性指標_exp": np.exp(0.05 * 50),
                    "知的生産性": 0,
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


def normalize_data(data: float, column_name: pd.Series) -> float:
    """
    各指標を正規化する関数

    Args:
        data (float): 正規化する値
        column_name (pd.Series): 正規化の基準となる値の列

    Returns:
        float: 正規化された値
    """
    # Check if max and min are the same (all values identical)
    denominator = column_name.max() - column_name.min()
    
    if denominator == 0 or pd.isna(denominator):
        # If all values are identical, return 0.5 (mid-point of normalization range)
        return 0.5
    else:
        # Normal normalization
        return (data - column_name.min()) / denominator


def extract_master_data(
    df: pd.DataFrame, time: int, master_data: pd.DataFrame
) -> pd.DataFrame:
    """
    月別に必要なマスタデータを抽出する関数
    
    This function now handles multi-row headers properly by looking at row values rather than header names

    Args:
        df (pd.DataFrame): データフレーム
        time (int): データフレームの各行のインデックス（タイムステップ）
        master_data (pd.DataFrame): マスタデータフレーム (with header=None)
    """
    try:
        # Get the current month
        current_month = df["date"][time].date().month
        
        # Check if master_data is empty
        if master_data.empty:
            print("Warning: Master data is empty")
            # Return an empty DataFrame with the right shape
            return pd.DataFrame(np.zeros((1, 30)))  # Return empty with 30 columns for all the expected data
            
        # Locate the month column (typically the first column)
        month_col_idx = 0
        
        # Find month header row (assuming it's in the first or second row)
        month_header_row = 0
        if "月" in master_data.iloc[0].values:
            month_header_row = 0
        elif "月" in master_data.iloc[1].values:
            month_header_row = 1
        
        # Identify the first data row (after headers)
        first_data_row = month_header_row + 1
        
        # Find the row for the current month
        month_rows = master_data[master_data.iloc[:, month_col_idx] == current_month]
        
        if month_rows.empty:
            print(f"No data found for month {current_month}, using first month's data")
            # Use the first month's data as a fallback
            month_data_row = master_data.iloc[first_data_row:first_data_row+1, :]
        else:
            month_data_row = month_rows.iloc[0:1, :]
        
        # Return just the data values for the month
        return month_data_row
        
    except Exception as e:
        print(f"Error in extract_master_data: {e}")
        # Return a minimal DataFrame with zeros
        return pd.DataFrame(np.zeros((1, 30)))  # Return empty with 30 columns for all the expected data