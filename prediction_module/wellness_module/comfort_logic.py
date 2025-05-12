import math

import numpy as np
import pandas as pd
from pythermalcomfort.models import pmv
from pythermalcomfort.utilities import v_relative


# 快適性指標の予測関数
def calculate_comfort(
    df: pd.DataFrame, column_name: str, master_com: pd.DataFrame, t: int
) -> float:
    """
    快適性指標の予測関数

    Args:
      df (pd.DataFrame): データフレーム
      column_name (str): 列名
      master_com (pd.DataFrame): マスタデータフレーム containing comfort parameters
      t (int): データフレームの各行のインデックス（タイムステップ）
    """

    df_cons = df.copy()
    df_cons["PMV"] = np.nan
    df_cons["PPD"] = np.nan
    df_cons["快適性指標"] = np.nan
    
    # Check if master_com has the required columns
    required_cols = ["室内相対湿度 [%]", "代謝量 [met]", "着衣量 [clo]", "気流速度 [m/s]"]
    if not all(col in master_com.columns for col in required_cols):
        # If not, use default values
        indoor_relative_humidity = 50  # Default relative humidity (%)
        met = 1.1  # Default metabolic rate
        clo = 0.8  # Default clothing insulation
        v = 0.1  # Default air velocity
        
        # Print warning
        missing = [col for col in required_cols if col not in master_com.columns]
        print(f"Warning: Missing columns in master_com: {missing}. Using default values.")
    else:
        # Use values from master_com
        indoor_relative_humidity = master_com.loc[0, "室内相対湿度 [%]"]
        met = master_com.loc[0, "代謝量 [met]"]
        clo = master_com.loc[0, "着衣量 [clo]"]
        v = master_com.loc[0, "気流速度 [m/s]"]

    vr = v_relative(
        v=v, met=1
    )  # metは代謝量だが，1に設定すると受領データとの当てはまりがよかったので1にしている

    try:
        # 更新部分
        # tr_column = [s for s in column_name if "設定温度" in s][0]
        tr_column = column_name
        
        # For t=0, use the same value for tdb and tr
        if t == 0:
            tdb = df.loc[t, tr_column]
        else:
            tdb = df.loc[t-1, tr_column]  # t=0 エラー出る、回避方法, t=0 tをよぶ
            
        tr = df.loc[t, tr_column]

        df_cons.loc[t, "PMV"] = pmv_value = pmv(
            tdb=tdb,
            tr=tr,
            vr=vr,
            rh=indoor_relative_humidity,
            met=met,
            clo=clo,
        )
        df_cons.loc[t, "PPD"] = 100 - 95 * pow(
            math.e,
            -(
                0.03353 * pow(df_cons.loc[t, "PMV"], 4)
                + 0.2719 * pow(df_cons.loc[t, "PMV"], 2)
            ),
        )
        df_cons.loc[t, "快適性指標"] = comfort_index = 100 - df_cons.loc[t, "PPD"]
    except Exception as e:
        print(f"Error calculating comfort: {e}")
        pmv_value = 0
        comfort_index = 50  # Neutral comfort as fallback
        indoor_relative_humidity = 50  # Default value

    return comfort_index, pmv_value, indoor_relative_humidity