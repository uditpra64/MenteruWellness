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
        master_pro (pd.DataFrame): マスタデータフレーム
        t (int): データフレームの各行のインデックス（タイムステップ）
    """

    df_cons = df.copy()
    df_cons["知的生産性"] = np.nan

    setting_t_column = column_name

    # Get hour
    hour = int(pd.to_datetime(df.loc[t, "datetime"]).time().strftime("%H"))
    
    # With header=1, columns are named like '知的生産性_朝_A', '知的生産性_朝_B', etc.
    # We can access them directly by name
    if hour >= 8 and hour <= 10:
        # Morning
        if "知的生産性_朝_A" in master_pro.columns:
            df_cons.loc[t, "知的生産性"] = (
                master_pro.loc[0, "知的生産性_朝_A"] * df.loc[t, setting_t_column] ** 2
                + master_pro.loc[0, "知的生産性_朝_B"] * df.loc[t, setting_t_column]
                + master_pro.loc[0, "知的生産性_朝_C"]
            )
    elif hour >= 13 and hour <= 15:
        # Afternoon
        if "知的生産性_昼_A" in master_pro.columns:
            df_cons.loc[t, "知的生産性"] = (
                master_pro.loc[0, "知的生産性_昼_A"] * df.loc[t, setting_t_column] ** 2
                + master_pro.loc[0, "知的生産性_昼_B"] * df.loc[t, setting_t_column]
                + master_pro.loc[0, "知的生産性_昼_C"]
            )
    elif hour >= 16 and hour <= 19:
        # Evening
        if "知的生産性_夕_A" in master_pro.columns:
            df_cons.loc[t, "知的生産性"] = (
                master_pro.loc[0, "知的生産性_夕_A"] * df.loc[t, setting_t_column] ** 2
                + master_pro.loc[0, "知的生産性_夕_B"] * df.loc[t, setting_t_column]
                + master_pro.loc[0, "知的生産性_夕_C"]
            )

    return df_cons.loc[t, "知的生産性"]