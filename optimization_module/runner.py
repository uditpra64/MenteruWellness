# optimization_module/runner.py
import logging
import os
import pandas as pd
from data_module.config.utils import (
    get_path_from_config,
    get_path_from_config_for_outside_base_folder,
)
from optimization_module.optimize_logic import run_optimization_logic
try:
    from prediction_module.utilities.models import XGBoostModel
except ImportError:
    XGBoostModel = object 

logger = logging.getLogger(__name__)

class OptimizationRunner:
    def __init__(
        self,
        input_features: list[str],
        temperature_setpoints_columns_dict: dict, 
        case_number: int,
        input_data_folder_path_key: str = "preprocessed_folder_path",
        input_data_file_name: str = "preprocessed_data.csv",
        start_study_date: str = "2021-04-01",
        end_study_date: str = "2023-03-31",
        start_optimize_date: str = "2023-04-01",
        end_optimize_date: str = "2023-04-30",
        master_data_path_key: str = "master_data_path",
        train_memory_flag: bool = False,
        lineage: str = "4F_西",
        model: XGBoostModel = None,
    ):
        self.input_features = input_features
        self.temperature_setpoints_columns_dict = temperature_setpoints_columns_dict
        self.case_number = case_number
        self.case_sheet_name = f"重み係数_Case{self.case_number}"

        self.input_data_folder_path = get_path_from_config_for_outside_base_folder(
            input_data_folder_path_key
        )
        self.input_data_file_name = input_data_file_name
        
        self.start_study_date = start_study_date
        self.end_study_date = end_study_date
        self.start_optimize_date = start_optimize_date
        self.end_optimize_date = end_optimize_date
        
        excel_path = get_path_from_config(master_data_path_key)
        self.master_data_excel_sheets = {}
        try:
            with pd.ExcelFile(excel_path) as xls:
                sheet_names = xls.sheet_names
                
                required_optim_sheet = "最適化"
                if required_optim_sheet not in sheet_names:
                    raise ValueError(f"Sheet '{required_optim_sheet}' not found in {excel_path}")
                self.master_data_excel_sheets[required_optim_sheet] = pd.read_excel(xls, sheet_name=required_optim_sheet, header=1) 
                logger.info("--- DEBUG RUNNER ---")
                logger.info("Columns of '最適化' sheet IN RUNNER: %s", self.master_data_excel_sheets[required_optim_sheet].columns.tolist())
                # logger.info("First 2 data rows of '最適化' sheet IN RUNNER:\n%s", self.master_data_excel_sheets[required_optim_sheet].head(2).to_string())
                logger.info("--------------------")
                if self.case_sheet_name not in sheet_names:
                    raise ValueError(f"Sheet '{self.case_sheet_name}' not found in {excel_path}")
                # For case sheets, header=1 means Excel row 2 is the header.
                # Data for weights starts from Excel row 3.
                self.master_data_excel_sheets[self.case_sheet_name] = pd.read_excel(xls, sheet_name=self.case_sheet_name, header=1)

        except FileNotFoundError:
            logger.error(f"Master data file not found: {excel_path}")
            raise
        except ValueError as ve:
            logger.error(f"Error processing master data sheets from {excel_path}: {ve}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error reading master data Excel file {excel_path}: {e}")
            raise
            
        self.train_memory_flag = train_memory_flag
        self.input_data = None
        self.lineage = lineage
        self.model = model
        if self.model is None:
            logger.warning(f"No model provided for OptimizationRunner (Lineage: {self.lineage}, Case: {self.case_number}).")

    def run(self):
        self._load_data()
        if self.input_data is None:
            logger.error(f"Input data not loaded for lineage {self.lineage}. Aborting optimization for Case {self.case_number}.")
            return

        if self.model is None: 
            logger.error(f"Cannot run optimization for {self.lineage}, Case {self.case_number} because the model is missing.")
            return

        run_optimization_logic(
            df=self.input_data,
            input_features_columns=self.input_features,
            temperature_setpoints_columns_dict=self.temperature_setpoints_columns_dict,
            lineage=self.lineage,
            start_study_date=self.start_study_date,
            end_study_date=self.end_study_date,
            start_optimize=self.start_optimize_date,
            end_optimize=self.end_optimize_date,
            model=self.model,
            master_data_excel_sheets=self.master_data_excel_sheets,
            case_number=self.case_number,
            case_sheet_name=self.case_sheet_name,
            train_memory_flag=self.train_memory_flag,
        )

    def _load_data(self):
        try:
            data_file_path = os.path.join(self.input_data_folder_path, self.input_data_file_name)
            
            if not os.path.exists(data_file_path):
                logger.warning(f"Target file {data_file_path} not found. Looking for most recent CSV containing 'preprocessed_data'.")
                csv_files = [
                    f for f in os.listdir(self.input_data_folder_path) if f.endswith(".csv") and "preprocessed_data" in f
                ]
                if not csv_files:
                    raise FileNotFoundError(f"No suitable preprocessed CSV files found in {self.input_data_folder_path}")
                
                csv_files.sort(key=lambda x: os.path.getmtime(os.path.join(self.input_data_folder_path, x)), reverse=True)
                data_file_path = os.path.join(self.input_data_folder_path, csv_files[0])
                logger.info(f"Using most recent suitable CSV file: {csv_files[0]}")

            logger.info(f"Loading data from: {data_file_path}")
            self.input_data = pd.read_csv(data_file_path, encoding="cp932")

            if "datetime" not in self.input_data.columns or "date" not in self.input_data.columns:
                 logger.warning(f"Loaded data from {data_file_path} is missing 'datetime' or 'date' column.")

        except FileNotFoundError as fnf_error:
            logger.error(f"Error in data loading: {fnf_error}")
            self.input_data = None
        except Exception as e:
            logger.error(f"An unexpected error occurred during data loading: {e}", exc_info=True)
            self.input_data = None