import logging
import os

import pandas as pd
from data_module.config.utils import (
    get_path_from_config,
    get_path_from_config_for_outside_base_folder,
)
from prediction_module.electricity_module.train_logic import run_train_logic

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class PredictionRunner:
    """A class to run the prediction pipeline

    Args:
        input_features (list[str]): The input features for the model
        output_feature_columns (list[str]): The output feature columns for the model
        output_feature (str): The output feature for the model
        input_data_folder_path (str): The path to the preprocessed data
        input_data_file_name (str): The input data file
        model_output_path (str): The path to the model output
    Methods:
        run: Run the full data training pipeline
    """

    def __init__(
        self,
        input_features: list[str],
        output_feature_columns: list[str],
        output_feature: str,
        input_data_folder_path: str = "preprocessed_folder_path",
        input_data_file_name: str = "preprocessed_data.csv",
        model_output_path: str = "pkl_path",
    ):
        self.input_features = input_features
        self.output_feature_columns = output_feature_columns
        self.output_feature = output_feature
        self.input_data_folder_path = get_path_from_config_for_outside_base_folder(
            input_data_folder_path
        )
        self.input_data_file_name = input_data_file_name
        self.model_output_path = get_path_from_config(model_output_path)

    def run(self, lineage: str):
        """Run the full data training pipeline"""

        self._load_input_data()
        model = run_train_logic(
            input_data=self.data_for_train,
            lineage=lineage,
            input_features_columns=self.input_features,
            output_feature_columns=self.output_feature_columns,
            output_feature=self.output_feature,
            model_output_path=self.model_output_path,
        )

        return model

    def _load_input_data(self):
        try:
            data_file_path = None

            # Look for preprocessed_data.csv which is the expected filename
            if os.path.exists(
                os.path.join(
                    self.input_data_folder_path, self.input_data_file_name
                )
            ):
                data_file_path = os.path.join(
                    self.input_data_folder_path, self.input_data_file_name
                )
                logging.info(f"Found target file: {self.input_data_file_name}")

            # If specific file not found, list available CSV files
            if data_file_path is None:
                csv_files = [
                    f
                    for f in os.listdir(self.input_data_folder_path)
                    if f.endswith(".csv")
                ]

                if not csv_files:
                    raise FileNotFoundError(
                        f"No CSV files found in {self.input_data_folder_path}"
                    )

                # Sort by modification time (newest first)
                csv_files.sort(
                    key=lambda x: os.path.getmtime(
                        os.path.join(self.input_data_folder_path, x)
                    ),
                    reverse=True,
                )

                # Use the most recent file
                data_file_path = os.path.join(
                    self.input_data_folder_path, csv_files[0]
                )
                logging.info(f"Using most recent CSV file: {csv_files[0]}")

            # Load the data file
            logging.info(f"Loading data from: {data_file_path}")
            self.data_for_train = pd.read_csv(data_file_path, encoding="cp932")
            self.data_for_test = (
                self.data_for_train.copy()
            )  # Use same data for test since we'll split it later

            # Check for NaN values in critical columns
            if "datetime" in self.data_for_train.columns:
                logging.info(
                    f"NaN in datetime: {self.data_for_train['datetime'].isna().sum()}"
                )
            if "date" in self.data_for_train.columns:
                logging.info(f"NaN in date: {self.data_for_train['date'].isna().sum()}")
        except Exception as e:
            logging.error(f"Error in data loading or training: {e}")
            raise
