import numpy as np
import optuna
import pandas as pd
from prediction_module.utilities.models import XGBoostModel


class ScoreEvaluator:
    """スコア算出用クラス
    - MAE
    - MAPE
    - MSE
    - RMSE
    を算出。
    """

    def __init__(self, X: pd.DataFrame, true: np.array, pred: np.array):
        self.df = X.copy()
        self.df["true"] = true
        self.df["pred"] = pred

    def evaluate(self, is_print: bool = True):
        mae = self._calc_mae(is_print)
        rmse = self._calc_rmse(is_print)
        mape = self._calc_mape(is_print)
        mape_eps0 = self._calc_mape(eps=0, exclude_under=2, is_print=is_print)

        return mae, rmse, mape, mape_eps0

    def _calc_mae(self, is_print: bool = True) -> None:
        """平均絶対誤差(Mean Absolute Error)を算出。
        MAE = Σ(|true - pred|) / N
        N: サンプル数
        """
        error = self.df["true"].values - self.df["pred"].values
        self.df["true-pred"] = error

        absolute_error = np.abs(error)
        self.df["abs(true-pred)"] = absolute_error
        self.mean_absolute_error = np.mean(absolute_error)
        if is_print:
            print(f"MAE = {self.mean_absolute_error}")
        return self.mean_absolute_error

    def _calc_mape(
        self, is_print: bool = True, eps: float = 1e-3, exclude_under: float = None
    ) -> None:
        """平均絶対誤差率(Mean Absolute Error)を算出。
        true = true[true >= exclude_under]
        pred = pred[true >= exclude_under]
        MAPE = Σ((|true + eps - pred|) / true + eps) / N
        N: サンプル数
        eps: 補正項 (trueが小さい場合に補正するための項)
        exclude_under: 評価対象とするデータの最小値。この値を下回るデータを評価には使用しない。
        """
        use_idxs = [i for i in range(self.df.shape[0])]
        true = self.df["true"].values
        pred = self.df["pred"].values
        self.df_ape = self.df.copy()
        if exclude_under is not None:
            use_idxs = np.where(true >= exclude_under)[0]
            true = true[use_idxs]
            pred = pred[use_idxs]

            self.df_ape = self.df_ape.iloc[use_idxs]

        percentage_error = (true + eps - pred) / (true + eps)
        self.df["pct(true-pred)"] = np.nan
        self.df["pct(true-pred)"].iloc[use_idxs] = percentage_error * 100
        self.absolute_percentage_error = np.abs(percentage_error)
        self.df["abs(pct(true-pred))"] = np.nan
        self.df["abs(pct(true-pred))"].iloc[use_idxs] = (
            self.absolute_percentage_error * 100
        )
        self.mean_absolute_percentage_error = (
            np.mean(self.absolute_percentage_error) * 100
        )
        if is_print:
            print(
                f"MAPE(%) --eps={eps} --exclude_under={exclude_under} = {self.mean_absolute_percentage_error}"
            )

        self.df_ape["absolute_percentage_error(%)"] = (
            self.absolute_percentage_error * 100
        )
        return self.mean_absolute_percentage_error

    def _calc_mse(self, is_print: bool = True) -> None:
        """平均二乗誤差(Mean Squared Error)を算出。
        MSE = Σ((true - pred)^2) / N
        N: サンプル数
        """
        error = self.df["true"].values - self.df["pred"].values
        squared_error = np.square(error)
        self.mean_squared_error = np.mean(squared_error)
        if is_print:
            print(f"MSE = {self.mean_squared_error}")
        return self.mean_squared_error

    def _calc_rmse(self, is_print: bool = True) -> None:
        """平均二乗平方根誤差(Root Mean Squared Error)を算出。
        RMSE = sqrt (Σ((true - pred)^2) / N )
        N: サンプル数
        """
        self._calc_mse(is_print=False)
        self.root_mean_squared_error = np.sqrt(self.mean_squared_error)
        if is_print:
            print(f"RMSE = {self.root_mean_squared_error}")
        return self.root_mean_squared_error


# パラメータチューニングに用いる関数
def objective(
    trial: optuna.Trial,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    input_features: list[str],
    output_feature: str,
    params: dict,
    model_name: str,
) -> float:
    """Optunaのハイパーパラメータチューニングに用いる関数

    Args:
        trial (optuna.Trial): Optunaの試行
        train (pd.DataFrame): 学習データ
        valid (pd.DataFrame): テストデータ
        input_features (list[str]): 説明変数
        output_feature (str): 目的変数
        params (dict): ハイパーパラメータの辞書
        model_name (str): モデル名
    """
    # Define the hyperparameters to tune
    params = params.copy()

    if model_name == "xgboost":
        model_tuned = XGBoostModel(input_features, output_feature, params)

    X_train, y_train, X_val, y_val = model_tuned.preprocess(train, valid)

    model_tuned.train(X_train, y_train)

    pred_val = model_tuned.predict(X_val)

    # Evaluate the model_tuned
    try:
        score_evaluator_val = ScoreEvaluator(X_val, y_val, pred_val)
        val_score = score_evaluator_val.evaluate(is_print=False)
        print(f"Trial {trial.number}, Val score: {val_score}")
        if val_score is None or not isinstance(val_score, (int, float)):
            raise ValueError("Invalid evaluation score")

    except Exception as e:
        print(f"Trial failed due to an error: {e}")
        return float("inf")  # Return a default bad score

    return val_score


def print_study_summaries(storage_url: str) -> None:
    """
    指定されたOptunaストレージに保存されているすべてのスタディの概要を印刷します。

    パラメータ:
    - storage_url (str): OptunaストレージへのURL。
    """
    # ストレージからすべてのスタディの概要を取得します
    study_summaries = optuna.get_all_study_summaries(storage_url)

    # 各スタディの概要を反復処理し、詳細を印刷します
    for study_summary in study_summaries:
        print("スタディ名:", study_summary.study_name)
        print("  試行回数:", study_summary.n_trials)

        # ベストトライアルにアクセスするためにスタディをロードします
        study = optuna.load_study(
            study_name=study_summary.study_name, storage=storage_url
        )
        best_trial = study.best_trial

        print("  ベストトライアル番号:", best_trial.number)
        print("  ベスト値(MAE):", best_trial.value)
        print("  ベストパラメータ:", best_trial.params)
        print("----------")  # スタディ間の可読性を高めるための区切り文字


def find_best_study(storage_url: str) -> None:
    """
    指定されたストレージURLから最良のスタディを見つけて情報を出力する関数です。

    Parameters:
    - storage_url (str): スタディが保存されているOptunaのストレージURL。
    """

    # 最良のスタディを追跡するための変数を初期化します
    best_study_name = None
    best_study_value = float(
        "inf"
    )  # 最小化の場合はfloat('inf')、最大化の場合は-float('inf')を使用します
    best_study_params = None
    best_study_trial_number = None

    # ストレージからすべてのスタディの概要を取得します
    all_study_summaries = optuna.study.get_all_study_summaries(storage=storage_url)

    # 各スタディの概要を反復処理します
    for study_summary in all_study_summaries:
        # ベストトライアルにアクセスするためにスタディをロードします
        study = optuna.load_study(
            study_name=study_summary.study_name, storage=storage_url
        )

        # 現在のスタディのベストトライアルをこれまでの最良のものと比較します
        if (
            study.best_trial.value < best_study_value
        ):  # 最小化の場合は'<'を使用し、最大化の場合は'>'を使用します
            best_study_name = study.study_name
            best_study_value = study.best_trial.value
            best_study_params = study.best_trial.params
            best_study_trial_number = study.best_trial.number

    # 最良のスタディの情報を出力します
    print("ベストスタディ名:", best_study_name)
    print("ベストトライアル番号:", best_study_trial_number)
    print("ベスト値(MAE):", best_study_value)
    print("ベストハイパーパラメータ:", best_study_params)

    return best_study_params
