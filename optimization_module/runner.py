import logging
import os

import pandas as pd
from data_module.config.utils import (
    get_path_from_config,
    get_path_from_config_for_outside_base_folder,
)
from optimization_module.optimize_logic import run_optimization_logic
from prediction_module.utilities.models import XGBoostModel

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class OptimizationRunner:
    """
    Runs one lineage through all five 重み係数_CaseX sheets.
    """

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
        full_master_sheet: str = "最適化",
        lineage: str = "4F_西",
        model: XGBoostModel = None,
        train_memory_flag: bool = False,
    ):
        self.input_features                = input_features
        self.temperature_setpoints_columns = temperature_setpoints_columns
        self.input_data_folder_path        = get_path_from_config_for_outside_base_folder(
            input_data_folder_path
        )
        self.input_data_file_name          = input_data_file_name
        self.start_study_date              = start_study_date
        self.end_study_date                = end_study_date
        self.start_optimize_date           = start_optimize_date
        self.end_optimize_date             = end_optimize_date
        self.master_data_path              = master_data_path
        self.full_master_sheet             = full_master_sheet
        self.lineage                       = lineage
        self.model                         = model
        self.train_memory_flag             = train_memory_flag

        # Load the full sheet once
        master_full_path = get_path_from_config(master_data_path)
        self.master_data_full = pd.read_excel(
            master_full_path,
            sheet_name=full_master_sheet
        )

        # We'll loop and load each of these sheets in run()
        self.weight_sheets = [
            "重み係数_Case1",
            "重み係数_Case2",
            "重み係数_Case3",
            "重み係数_Case4",
            "重み係数_Case5",
        ]

        self.input_data = None

    def run(self):
        # Load your preprocessed CSV (no DB)
        self._load_data()

        master_full_path = get_path_from_config(self.master_data_path)

        for case_sheet in self.weight_sheets:
            logging.info(f"▶ Running optimization for {self.lineage} / {case_sheet}")
            # Load 4 weight coefficients
            weight_data = pd.read_excel(master_full_path, sheet_name=case_sheet)

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
                weight_data=weight_data,
                master_data_full=self.master_data_full,
                train_memory_flag=self.train_memory_flag,
            )

    def _load_data(self):
        try:
            target_path = os.path.join(
                self.input_data_folder_path, self.input_data_file_name
            )
            if not os.path.exists(target_path):
                csvs = sorted(
                    [f for f in os.listdir(self.input_data_folder_path) if f.endswith(".csv")],
                    key=lambda fn: os.path.getmtime(os.path.join(self.input_data_folder_path, fn)),
                    reverse=True,
                )
                if not csvs:
                    raise FileNotFoundError(f"No CSV in {self.input_data_folder_path}")
                target_path = os.path.join(self.input_data_folder_path, csvs[0])
                logging.info(f"Using most recent CSV file: {csvs[0]}")
            else:
                logging.info(f"Found target file: {self.input_data_file_name}")

            logging.info(f"Loading data from: {target_path}")
            self.input_data = pd.read_csv(target_path, encoding="cp932")

            for col in ["datetime", "date"]:
                if col in self.input_data.columns:
                    logging.info(f"NaN in {col}: {self.input_data[col].isna().sum()}")

        except Exception as e:
            logging.error(f"Error loading preprocessed data: {e}")
            raise
