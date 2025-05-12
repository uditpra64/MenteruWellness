import numpy as np
import pandas as pd


def calculate_productivity(
    df: pd.DataFrame, column_name: str, master_pro: pd.DataFrame, t: int
) -> float:
    """
    知的生産性の予測関数

    Args:
        df (pd.DataFrame): データフレーム
        column_name (str): 列名
        master_pro (pd.DataFrame): マスタデータフレーム with productivity parameters
        t (int): データフレームの各行のインデックス（タイムステップ）
    """

    df_cons = df.copy()
    df_cons["知的生産性"] = np.nan

    # 更新部分
    # setting_t_column = [s for s in column_name if "設定温度" in s][0]
    setting_t_column = column_name
    
    # Ensure master_pro has all required columns
    required_cols = [
        "知的生産性_朝_A", "知的生産性_朝_B", "知的生産性_朝_C",
        "知的生産性_昼_A", "知的生産性_昼_B", "知的生産性_昼_C",
        "知的生産性_夕_A", "知的生産性_夕_B", "知的生産性_夕_C"
    ]
    
    # Check if master_pro has the required columns
    if not all(col in master_pro.columns for col in required_cols):
        # If not, print a warning (but don't raise an error)
        missing = [col for col in required_cols if col not in master_pro.columns]
        print(f"Warning: Missing columns in master_pro: {missing}")
        return np.nan
    
    # 更新部分
    try:
        hour = int(pd.to_datetime(df.loc[t, "datetime"]).time().strftime("%H"))
        
        # Morning calculation (8:00-10:59)
        if 8 <= hour <= 10:
            df_cons.loc[t, "知的生産性"] = (
                master_pro.loc[0, "知的生産性_朝_A"] * df.loc[t, setting_t_column] ** 2
                + master_pro.loc[0, "知的生産性_朝_B"] * df.loc[t, setting_t_column]
                + master_pro.loc[0, "知的生産性_朝_C"]
            )
        
        # Afternoon calculation (13:00-15:59)
        elif 13 <= hour <= 15:
            df_cons.loc[t, "知的生産性"] = (
                master_pro.loc[0, "知的生産性_昼_A"] * df.loc[t, setting_t_column] ** 2
                + master_pro.loc[0, "知的生産性_昼_B"] * df.loc[t, setting_t_column]
                + master_pro.loc[0, "知的生産性_昼_C"]
            )
        
        # Evening calculation (16:00-19:59)
        elif 16 <= hour <= 19:
            df_cons.loc[t, "知的生産性"] = (
                master_pro.loc[0, "知的生産性_夕_A"] * df.loc[t, setting_t_column] ** 2
                + master_pro.loc[0, "知的生産性_夕_B"] * df.loc[t, setting_t_column]
                + master_pro.loc[0, "知的生産性_夕_C"]
            )
    except Exception as e:
        print(f"Error calculating productivity: {e}")
        df_cons.loc[t, "知的生産性"] = np.nan

    return df_cons.loc[t, "知的生産性"]