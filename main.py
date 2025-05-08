import argparse
import logging
import os
import sys
import time

# ensure src/ is on PYTHONPATH
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from config_settings.config_optimize import (
    END_OPTIMIZE_DATE_OPTIMIZE,
    END_STUDY_DATE_OPTIMIZE,
    START_OPTIMIZE_DATE_OPTIMIZE,
    START_STUDY_DATE_OPTIMIZE,
    TEMPERATURE_SETPOINTS_COLUMNS,
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

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def parse_arguments():
    """Specify the execution steps of the script."""
    parser = argparse.ArgumentParser(
        description="Specify the execution steps of the script."
    )
    parser.add_argument(
        "--run-preprocess", action="store_true", help="Execute data preprocessing."
    )
    parser.add_argument("--run-train", action="store_true", help="Execute training.")
    parser.add_argument(
        "--run-optimize", action="store_true", help="Execute optimization."
    )
    parser.add_argument("--save-results", action="store_true", help="Save results.")
    return parser.parse_args()


def main():
    """
    Main function to run all modules

    To run all steps: uv run main.py
    To run specific steps (e.g. preprocess and train only):
      uv run main.py --run-preprocess --run-train
    """
    start = time.time()

    # ---------------------------------------
    # Parse Arguments 引数をパース
    # ---------------------------------------
    args = parse_arguments()
    if any([args.run_preprocess, args.run_train, args.run_optimize, args.save_results]):
        run_preprocess = args.run_preprocess
        run_train = args.run_train
        run_optimize = args.run_optimize
        run_save_results = args.save_results
    elif args.run_optimize:
        # If only optimize is specified, training must also run
        run_preprocess, run_train, run_optimize, run_save_results = (
            False,
            True,
            True,
            False,
        )
    else:
        # If no flags, run everything
        run_preprocess, run_train, run_optimize, run_save_results = (
            True,
            True,
            True,
            True,
        )

    # ---------------------------------------
    # Run Preprocessing データ前処理を実行
    # ---------------------------------------
    if run_preprocess:
        logging.info("Data preprocessing is running...")
        preprocess_runner = DataPreprocessRunner(
            start_date_preprocess=START_TIME_PREPROCESS,
            end_date_preprocess=END_TIME_PREPROCESS,
        )
        # --- COMMENTED OUT: skip SQL/database fetch so you can run locally ---
        # preprocess_runner.run()

    logging.info("====================================")

    # Four “lineages” (zones) to loop over
    lineages_list = ["4F_西", "4F_東", "5F_西", "5F_東"]

    if run_train:
        for lineage in lineages_list:
            # --------------------------------
            # Run Prediction 予測を実行
            # --------------------------------
            logging.info(f"Prediction is running... for {lineage}")
            predict_runner = PredictionRunner(
                input_features=LINEAGES_INPUT_FEATURES_MAPPING[lineage],
                output_feature_columns=LINEAGES_OUTPUT_FEATURES_MAPPING[lineage],
                output_feature=LINEAGES_AGGREGATED_OUTPUT_FEATURE,
            )
            model = predict_runner.run(lineage=lineage)

            if run_optimize:
                logging.info(f"Optimization is running... for {lineage}")
                optimize_runner = OptimizationRunner(
                    input_features=LINEAGES_INPUT_FEATURES_MAPPING[lineage],
                    temperature_setpoints_columns=TEMPERATURE_SETPOINTS_COLUMNS[lineage],
                    start_study_date=START_STUDY_DATE_OPTIMIZE,
                    end_study_date=END_STUDY_DATE_OPTIMIZE,
                    start_optimize_date=START_OPTIMIZE_DATE_OPTIMIZE,
                    end_optimize_date=END_OPTIMIZE_DATE_OPTIMIZE,
                    lineage=lineage,
                    model=model,
                    train_memory_flag=False,
                    full_master_sheet="最適化",
                )
                optimize_runner.run()

            # --------------------------------------------
            # Save Results for each lineage ごとの結果を保存
            # --------------------------------------------
            if run_save_results:
                logging.info(f"Saving results for {lineage}...")
                # TODO: extract and implement your saving logic here

    logging.info("====================================")
    elapsed = int(time.time() - start)
    logging.info(
        f"All logic completed successfully in {elapsed // 60}m {elapsed % 60}s"
    )


if __name__ == "__main__":
    main()
