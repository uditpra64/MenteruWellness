import warnings

import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning, module="xgboost")
pd.options.mode.chained_assignment = None
from xgboost import XGBRegressor  # noqa: E402


class XGBoostModel:
    """
    1日階差を取ったデータを使ってXGBoostモデルをトレーニングし、予測を行います。
    """

    def __init__(self, input_features, output_features, params={}):
        self.input_features = input_features
        self.output_features = output_features
        self.params = params
        self.model = XGBRegressor(**self.params)
        self.pred_min = 0
        self.pred_max = None

    def preprocess(self, train, valid):
        """
        トレーニングとバリデーションデータセットの前処理を行い、モデル用の特徴量を準備します。
        引数:
            train (DataFrame): トレーニングデータセット。
            valid (DataFrame): バリデーションデータセット。

        概要:
            - `train` から入力(`self.input_features`)と出力特徴量(`self.output_features`)を選択して、`X_train` と `y_train` を生成。
            - `valid` から同様に、`X_val` と `y_val` を生成。

        戻り値:
            (X_train, y_train, X_val, y_val): 各データセットの入力と出力特徴量のタプル。
        """
        X_train = train[self.input_features]
        y_train = train[self.output_features]
        X_val = valid[self.input_features]
        y_val = valid[self.output_features]
        return X_train, y_train, X_val, y_val

    def train(
        self,
        X,
        y,
    ):
        """
        トレーニングデータセットを使ってモデルをトレーニングします。

        引数:
            X (DataFrame): トレーニングデータセットの入力特徴量。
            y (DataFrame): トレーニングデータセットの出力特徴量。
        概要:
            - `X` と `y` を使ってモデルをトレーニング。
        """
        self.model.fit(X, y)

    def predict(self, X):
        """
        モデルを使って入力特徴量から出力特徴量を予測します。
        引数:
            X (DataFrame): 予測に使う入力特徴量。
        戻り値:
            (DataFrame): 予測された出力特徴量。
        """
        pred = self.model.predict(X)
        return pred

    def get_input_output_features(self, deep=True):
        """
        モデルの入力特徴量と出力特徴量を取得します。
        引数:
            deep (bool): Trueの場合、入力特徴量と出力特徴量のコピーを返します。
        戻り値:
            (dict): 入力特徴量と出力特徴量。
        """
        return {
            "input_features": self.input_features,
            "output_features": self.output_features,
            **self.params,
        }

    def get_default_params(self):
        """
        モデルのデフォルトパラメータを取得します。
        戻り値:
            (dict): デフォルトパラメータ。
        """
        default_params = XGBRegressor().get_params()
        return default_params
        # return self.model.get_params()

    def get_feature_importance(self):
        """
        モデルの特徴量の重要度を取得します。
        戻り値:
            (dict): 特徴量の重要度。
        """
        booster = self.model.get_booster()
        feature_importances = booster.get_score(importance_type="weight")
        return feature_importances
