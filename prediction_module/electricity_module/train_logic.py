import logging
import pickle
import time
import warnings

import optuna
import pandas as pd
from prediction_module.utilities.data_preprocess import (
    add_target_feature,
    preprocess_data,
    split_train_test,
)
from prediction_module.utilities.models import XGBoostModel
from prediction_module.utilities.score_utils import (
    ScoreEvaluator,
    find_best_study,
    objective,
    print_study_summaries,
)

# Suppress specific warnings
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)
warnings.filterwarnings("ignore", category=pd.errors.ParserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def run_train_logic(
    input_data: pd.DataFrame,
    lineage: str,
    input_features_columns: list[str],
    output_feature_columns: list[str],
    output_feature: str,
    model_output_path: str,
) -> XGBoostModel:
    """A function to run the training logic

    - Preprocess and Prepare Data: Preprocess and prepare the data for training and testing
    - Split Data into Training and Test Sets: Split the data into training and test sets
    - Add Objective (Target) Feature: Add the objective (target) feature to the training and test data
    - Hyperparameter Tuning: Hyperparameter tuning
    - Train Model: Train the model
    - Save Model: Save the model

    Args:
        input_data (pd.DataFrame): Data for training
        lineage (str): Lineage (e.g., "4F_西")
    """
    start = time.time()  # 現在時刻（処理開始前）を取得

    logging.info("input_data columns: %s", input_data.columns)
    logging.info("input_features_columns: %s", input_features_columns)
    logging.info("output_feature_columns: %s", output_feature_columns)
    logging.info("output_feature: %s", output_feature)

    # ------------------------------------------------
    # Preprocess and Prepare Data データの前処理と準備
    # ------------------------------------------------
    prepared_data = preprocess_data(input_data, lineage)
    data_train, data_test = split_train_test(prepared_data)
    data_train, data_test = add_target_feature(
        data_train, data_test, output_feature_columns, output_feature=output_feature
    )

    # ------------------------------------------------
    # Hyperparameter Tuning ハイパーパラメータチューニング
    # ------------------------------------------------
    storage_url = "sqlite:///wellness_study.db"
    study = optuna.create_study(
        study_name="wellness_study",
        storage=storage_url,
        direction="minimize",
        load_if_exists=True,  # If study exists, load it
    )
    study.optimize(
        lambda trial: objective(
            trial=trial,
            train=data_train,
            valid=data_test,
            input_features=input_features_columns,
            output_feature=output_feature,
            params={
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "max_delta_step": trial.suggest_int("max_delta_step", 0, 10),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            },
            model_name="xgboost",
        ),
        n_trials=50,
    )

    print_study_summaries(
        storage_url
    )  # Optunaストレージのすべてのスタディ概要を印刷します。
    best_params = find_best_study(storage_url)  # 最良のハイパーパラメータを取得

    # ---------------------------------------
    # Train Model モデルの訓練
    # ---------------------------------------
    model = XGBoostModel(
        input_features=input_features_columns,
        output_features=output_feature,
        params=best_params,
    )
    X_train, y_train, X_val, y_valid = model.preprocess(data_train, data_test)
    model.train(X_train, y_train)

    # ---------------------------------------
    # Predict Model モデルの予測
    # ---------------------------------------
    y_pred_train = model.predict(X_train)
    y_pred_valid = model.predict(X_val)
    score_evaluator_train = ScoreEvaluator(X_train, y_train, y_pred_train)
    score_evaluator_valid = ScoreEvaluator(X_val, y_valid, y_pred_valid)
    logging.info("Train Evaluation: ")
    score_evaluator_train.evaluate(is_print=True)
    logging.info("Validation Evaluation: ")
    score_evaluator_valid.evaluate(is_print=True)

    # ---------------------------------------
    # Save Model モデルの保存
    # ---------------------------------------
    filename = model_output_path + f"/predicted_electricity_{lineage}.pkl"
    logging.info(f"モデルを保存します: {filename}")
    with open(filename, "wb") as f:
        pickle.dump(model, f)  # electircity_moduleに保存

    end = time.time()  # 現在時刻（処理完了後）を取得
    train_time = end - start  # 処理完了後の時刻から処理開始前の時刻を減算する

    logging.info("学習が完了しました 時間={}秒".format(train_time))

    return model
