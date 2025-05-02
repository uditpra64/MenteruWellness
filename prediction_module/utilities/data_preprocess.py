import random

import pandas as pd


def preprocess_data(
    data: pd.DataFrame,
    lineage: str,
) -> pd.DataFrame:
    """データの準備
    - 必要データ(現在を起点とした3年前までのデータ)の抽出
    - 外れ値の削除
    - 学習用のデータのみ適用
    """
    # object型となっているdatetimeとdateをdatetime型に変更
    data["datetime"] = pd.to_datetime(data["datetime"])
    data["date"] = pd.to_datetime(data["date"])
    ##外れ値の削除
    # 室内温度
    floor = lineage.split("_")[0].split("F")[0]
    direction = lineage.split("_")[1]
    a = (
        data["室内温度_C_{}F執務室_{}".format(floor, direction)].describe().tolist()[1]
        + 2
        * data["室内温度_C_{}F執務室_{}".format(floor, direction)]
        .describe()
        .tolist()[2]
    )
    b = (
        data["室内温度_C_{}F執務室_{}".format(floor, direction)].describe().tolist()[1]
        - 2
        * data["室内温度_C_{}F執務室_{}".format(floor, direction)]
        .describe()
        .tolist()[2]
    )
    data = data[data["室内温度_C_{}F執務室_{}".format(floor, direction)] <= a]
    data = data[data["室内温度_C_{}F執務室_{}".format(floor, direction)] >= b]

    #
    # 空調電力(負荷)予測
    a = data["空調電力予測_kWh"].copy()
    b = a.describe().tolist()[1] + 2 * a.describe().tolist()[2]
    data = data[data["空調電力予測_kWh"] <= b]
    data_a = data[
        (data["hour"] >= 8) & (data["hour"] <= 18)
    ]  # コアタイム(8~18)のみ抽出
    a = data_a["空調負荷予測_kWh"].copy()
    b = a.describe().tolist()[1] + 2 * a.describe().tolist()[2]
    data_a = data_a[data_a["空調負荷予測_kWh"] <= b]
    max_air_load = data_a["空調負荷予測_kWh"].max()  # 最大負荷値算出
    data = data[data["空調負荷予測_kWh"] <= max_air_load]
    ##曜日データのダミー変数化
    data = pd.get_dummies(data, dtype=int)
    # インデックスの振り直し
    prepared_data = data.reset_index(drop=True)
    return prepared_data


def split_train_test(
    prepared_data: pd.DataFrame,
    train_fraction: float = 0.7,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split Data into Training and Test Sets"""
    n_train = int(len(prepared_data) * train_fraction)

    # Randomly sample indices for training split
    train_indices = random.sample(range(len(prepared_data)), k=n_train)

    # Get train datetimes from the prepared train data
    train_datetimes = prepared_data.loc[train_indices, "datetime"].tolist()

    # Remove overlapping datetimes from the test data
    test_datetimes = list(
        set(prepared_data["datetime"].tolist()) - set(train_datetimes)
    )

    # Create final train and test splits
    train_split = prepared_data.iloc[train_indices].copy()
    test_split = prepared_data[prepared_data["datetime"].isin(test_datetimes)].copy()

    return train_split, test_split


def add_target_feature(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    output_feature_columns: list[str],
    output_feature: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add Objective (Target) Feature to the training and test data"""

    train_dataframe = train_data.copy()
    test_dataframe = test_data.copy()

    # Fill missing values with 0
    train_dataframe[output_feature_columns[0]] = train_dataframe[
        output_feature_columns[0]
    ].fillna(0)
    train_dataframe[output_feature_columns[1]] = train_dataframe[
        output_feature_columns[1]
    ].fillna(0)
    test_dataframe[output_feature_columns[0]] = test_dataframe[
        output_feature_columns[0]
    ].fillna(0)
    test_dataframe[output_feature_columns[1]] = test_dataframe[
        output_feature_columns[1]
    ].fillna(0)

    # Calculate and assign the target variable for training data
    train_dataframe[output_feature] = (
        train_dataframe[output_feature_columns[0]]
        + train_dataframe[output_feature_columns[1]]
    ).tolist()

    # Calculate and assign the target variable for test data
    test_dataframe[output_feature] = (
        test_dataframe[output_feature_columns[0]]
        + test_dataframe[output_feature_columns[1]]
    ).tolist()

    return train_dataframe, test_dataframe
