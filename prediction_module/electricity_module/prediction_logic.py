import pandas as pd
from prediction_module.utilities.models import XGBoostModel


def run_prediction_logic(
    data: pd.DataFrame,
    used_column_list: list[str],
    time: int,
    model: XGBoostModel,
) -> float | None:
    """消費電力量の予測関数"""
    if model is None:
        return None  # モデルが存在しない場合に処理を中断

    data = data[
        data.columns[data.columns.isin(used_column_list)]
    ]  # テストデータの説明変数を抽出
    data = data.iloc[time : time + 1, :]  # 時間ごとのデータを取得

    try:
        predictions = model.predict(data)
        prediction = predictions[0]
    except Exception as e:
        print(f"予測エラー: {e}")
        return None

    # 予測結果が負になった場合は0にする
    if prediction < 0:
        prediction = 0

    # 予測対象データのON/OFFが0の場合は0にする
    if data.iloc[0, 8] == 0:
        prediction = 0

    return prediction
