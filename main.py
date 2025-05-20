# main.py
import argparse
import logging
import os
import sys
import time
import pandas as pd

# sys.path.append(os.path.join(os.path.dirname(__file__), "src")) # Only if you have a src folder

from config_settings.config_optimize import (
    END_OPTIMIZE_DATE_OPTIMIZE,
    END_STUDY_DATE_OPTIMIZE,
    START_OPTIMIZE_DATE_OPTIMIZE,
    START_STUDY_DATE_OPTIMIZE,
    TEMPERATURE_SETPOINTS_COLUMNS, # This IS a dict of {"lineage": "column_name"}
)
from config_settings.config_predict import (
    LINEAGES_AGGREGATED_OUTPUT_FEATURE,
    LINEAGES_INPUT_FEATURES_MAPPING,
    LINEAGES_OUTPUT_FEATURES_MAPPING,
)
from config_settings.config_preprocess import END_TIME_PREPROCESS, START_TIME_PREPROCESS
from data_module.runner import DataPreprocessRunner
from optimization_module.runner import OptimizationRunner
from prediction_module.runner import PredictionRunner
try:
    from prediction_module.utilities.models import XGBoostModel
except ImportError:
    XGBoostModel = object # Placeholder for type hinting if models.py is complex
from data_module.config.utils import get_path_from_config

logger = logging.getLogger(__name__)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Specify the execution steps of the script.")
    parser.add_argument("--run-preprocess", action="store_true", help="Execute data preprocessing.")
    parser.add_argument("--run-train", action="store_true", help="Execute training.")
    parser.add_argument("--run-optimize", action="store_true", help="Execute optimization.")
    parser.add_argument(
        "--lineages", type=str, default="4F_西,4F_東,5F_西,5F_東",
        help="Comma-separated list of lineages to process (e.g., 4F_西,5F_東)."
    )
    parser.add_argument(
        "--cases", type=str, default="1,2,3,4,5",
        help="Comma-separated list of case numbers to run for optimization (e.g., 1,3,5)."
    )
    return parser.parse_args()

def main():
    start_main_time = time.time()
    args = parse_arguments()

    run_all = not any([args.run_preprocess, args.run_train, args.run_optimize])
    run_preprocess_flag = args.run_preprocess or run_all
    run_train_flag = args.run_train or run_all
    run_optimize_flag = args.run_optimize or run_all

    lineages_to_process = [lin.strip() for lin in args.lineages.split(',')]
    cases_to_process = [int(c.strip()) for c in args.cases.split(',')]

    if run_preprocess_flag:
        logger.info("Data preprocessing is running...")
        preprocess_runner = DataPreprocessRunner(
            start_date_preprocess=START_TIME_PREPROCESS,
            end_date_preprocess=END_TIME_PREPROCESS,
        )
        preprocess_runner.run()
    logger.info("====================================")

    trained_models: dict[str, XGBoostModel | None] = {}

    if run_train_flag:
        for lineage in lineages_to_process:
            logger.info(f"Prediction model training is running for lineage: {lineage}")
            if lineage not in LINEAGES_INPUT_FEATURES_MAPPING or \
               lineage not in LINEAGES_OUTPUT_FEATURES_MAPPING:
                logger.error(f"Configuration for lineage {lineage} not found in config_predict.py. Skipping training.")
                trained_models[lineage] = None
                continue
            try:
                predict_runner = PredictionRunner(
                    input_features=LINEAGES_INPUT_FEATURES_MAPPING[lineage],
                    output_feature_columns=LINEAGES_OUTPUT_FEATURES_MAPPING[lineage],
                    output_feature=LINEAGES_AGGREGATED_OUTPUT_FEATURE,
                )
                model = predict_runner.run(lineage=lineage)
                trained_models[lineage] = model
            except Exception as e:
                logger.error(f"Error during training for lineage {lineage}: {e}", exc_info=True)
                trained_models[lineage] = None
            logger.info(f"==================================== (End Train {lineage})")
    else:
        logger.info("Skipping training. Attempting to load pre-trained models for optimization.")
        model_base_path = get_path_from_config("pkl_path")
        for lineage in lineages_to_process:
            model_file = os.path.join(model_base_path, f"predicted_electricity_{lineage}.pkl")
            if os.path.exists(model_file):
                try:
                    with open(model_file, "rb") as f:
                        trained_models[lineage] = pd.read_pickle(f)
                    logger.info(f"Successfully loaded model for lineage {lineage} from {model_file}")
                except Exception as e:
                    logger.error(f"Error loading model for lineage {lineage} from {model_file}: {e}", exc_info=True)
                    trained_models[lineage] = None
            else:
                logger.warning(f"Pre-trained model for lineage {lineage} not found at {model_file}.")
                trained_models[lineage] = None

    if run_optimize_flag:
        for lineage in lineages_to_process:
            model_to_use = trained_models.get(lineage)
            if model_to_use is None:
                logger.warning(f"No model available for lineage {lineage}. Skipping optimization for this lineage.")
                continue
            if lineage not in TEMPERATURE_SETPOINTS_COLUMNS:
                logger.error(f"Temperature setpoint column key for lineage {lineage} not found in config_optimize.TEMPERATURE_SETPOINTS_COLUMNS. Skipping.")
                continue
            if lineage not in LINEAGES_INPUT_FEATURES_MAPPING:
                logger.error(f"Input features for lineage {lineage} not found in config_predict.LINEAGES_INPUT_FEATURES_MAPPING. Skipping.")
                continue

            for case_num in cases_to_process:
                logger.info(f"Optimization is running for lineage: {lineage}, Case: {case_num}")
                try:
                    optimize_runner = OptimizationRunner(
                        input_features=LINEAGES_INPUT_FEATURES_MAPPING[lineage],
                        temperature_setpoints_columns_dict=TEMPERATURE_SETPOINTS_COLUMNS,
                        case_number=case_num,
                        start_study_date=START_STUDY_DATE_OPTIMIZE,
                        end_study_date=END_STUDY_DATE_OPTIMIZE,
                        start_optimize_date=START_OPTIMIZE_DATE_OPTIMIZE,
                        end_optimize_date=END_OPTIMIZE_DATE_OPTIMIZE,
                        lineage=lineage,
                        model=model_to_use,
                        train_memory_flag=False, # Default to False to regenerate df_for_normalize
                    )
                    optimize_runner.run()
                except Exception as e:
                    logger.error(f"Error during optimization for lineage {lineage}, Case {case_num}: {e}", exc_info=True)
                logger.info(f"==================================== (End Optimize {lineage} Case {case_num})")

    logger.info("====================================")
    conducted_time = int(time.time() - start_main_time)
    logger.info(
        "All requested operations completed in {}m {}s".format(
            conducted_time // 60, conducted_time % 60
        )
    )

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s")
    main()