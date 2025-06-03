import logging
import os

import pandas as pd
from data_module.config.utils import (
    get_path_from_config,
    get_path_from_config_for_outside_base_folder,
)
from optimization_module.optimize_logic import run_optimization_logic
from prediction_module.electricity_module.train_logic import XGBoostModel

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class OptimizationRunner:
    def __init__(
        self,
        input_features: list[str],
        temperature_setpoints_columns: list[str],
        input_data_folder_path: str = "preprocessed_folder_path",
        input_data_file_name: str = "preprocessed_data.csv",
        start_study_date: str = "2021-04-01",
        end_study_date: str = "2023-03-31",
        start_optimize_date: str = "2023-04-01",
        end_optimize_date: str = "2023-04-30",
        master_data_path: str = "master_data_path",
        train_memory_flag: bool = False,
        lineage: str = "4F_西",
        model: XGBoostModel = None,
        case_num: int = 1,  # Add case number parameter
    ):
        self.input_features = input_features
        self.temperature_setpoints_columns = temperature_setpoints_columns
        self.input_data_folder_path = get_path_from_config_for_outside_base_folder(
            input_data_folder_path
        )
        self.input_data_file_name = input_data_file_name
        self.start_study_date = start_study_date
        self.end_study_date = end_study_date
        self.start_optimize_date = start_optimize_date
        self.end_optimize_date = end_optimize_date
        
        # Read master data with header=1 to skip merged cells
        master_data_file = get_path_from_config(master_data_path)
        self.master_data = pd.read_excel(master_data_file, sheet_name=None, header=1)
        
        self.train_memory_flag = train_memory_flag
        self.input_data = None
        self.lineage = lineage
        self.model = model
        self.case_num = case_num  # Store case number

    def run(self):
        self._load_data()
        run_optimization_logic(
            df=self.input_data,
            input_features_columns=self.input_features,
            temperature_setpoints_columns=self.temperature_setpoints_columns,
            lineage=self.lineage,
            start_study_date=self.start_study_date,
            end_study_date=self.end_study_date,
            start_optimize=self.start_optimize_date,
            end_optimize=self.end_optimize_date,
            model=self.model,
            master_data=self.master_data,
            train_memory_flag=self.train_memory_flag,
            case_num=self.case_num,  # Pass case number to optimization logic
        )

    def _load_data(self):
        try:
            data_file_path = None

            # Look for preprocessed_data.csv which is the expected filename
            if os.path.exists(
                os.path.join(self.input_data_folder_path, self.input_data_file_name)
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
                data_file_path = os.path.join(self.input_data_folder_path, csv_files[0])
                logging.info(f"Using most recent CSV file: {csv_files[0]}")

            # Load the data file
            logging.info(f"Loading data from: {data_file_path}")
            self.input_data = pd.read_csv(data_file_path, encoding="cp932")

            # Check for NaN values in critical columns
            if "datetime" in self.input_data.columns:
                logging.info(
                    f"NaN in datetime: {self.input_data['datetime'].isna().sum()}"
                )
            if "date" in self.input_data.columns:
                logging.info(f"NaN in date: {self.input_data['date'].isna().sum()}")
        except Exception as e:
            logging.error(f"Error in data loading or training: {e}")
            raise