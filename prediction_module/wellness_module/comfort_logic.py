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
      master_com (pd.DataFrame): マスタデータフレーム
      t (int): データフレームの各行のインデックス（タイムステップ）
    """

    df_cons = df.copy()
    df_cons["PMV"] = np.nan
    df_cons["PPD"] = np.nan
    df_cons["快適性指標"] = np.nan

    indoor_relative_humidity = master_com.iloc[0, 0]
    met = master_com.iloc[0, 1]
    clo = master_com.iloc[0, 2]
    v = master_com.iloc[0, 3]

    vr = v_relative(
        v=v, met=1
    )  # metは代謝量だが，1に設定すると受領データとの当てはまりがよかったので1にしている

    # 更新部分
    # tr_column = [s for s in column_name if "設定温度" in s][0]
    tr_column = column_name
    tdb = df.loc[
        t - 1 if t > 0 else t, tr_column
    ]  # t= 0 エラー出る、回避方法, t=0 tをよぶ
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

    return comfort_index, pmv_value, indoor_relative_humidity
