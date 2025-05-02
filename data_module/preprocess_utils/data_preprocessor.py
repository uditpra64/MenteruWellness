import logging
import os
import sys
import time
import traceback
from datetime import datetime

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from config_settings.config_preprocess import (
    MIN_MAX_TABLE_NAMES,
    PREPROCESS_COLUMN_MAPPING,
    PREPROCESS_FINAL_COLS,
)
from data_module.config.utils import get_path_from_config_for_outside_base_folder
from data_module.preprocess_utils.utils import (
    add_pre_timestep_data,
    adjust_hourly_data_for_period,
    get_weekday,
    on_off_judge_for_predict,
    operation_mode_judge,
    wetbulb_temp,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class DataPreprocessor:
    """
    A class to handle the preprocessing of data for energy consumption analysis.

    This class is responsible for loading configuration files, processing raw data into an hourly format,
    adjusting the data based on specified date ranges, and preparing datasets for training, testing, and prediction.
    It includes methods for renaming columns, assigning relevant features, and saving the preprocessed data to CSV files.

    Attributes:
        REQUIRED_COLUMNS (set): A set of required column names for preprocessing configuration.
        PREPROCESSED_REQUIRED_COLUMNS (list): A list of required columns for the preprocessed data.
        FLOOR_TEMP_COLUMN_MAPPINGS (dict): A mapping of new column names to their corresponding start and end indices in the original dataframe.
        OFFICE_ZONES (list): A list of office zones for which data is processed.
        OFFICE_FLOORS (list): A list of office floors for which data is processed.
        ROOM_DIRECTIONS (list): A list of room directions (e.g., "西", "東").

    Methods:
        _load_preprocessing_config(): Loads configuration files needed for preprocessing.
        _filter_preprocessing_files(): Filters the list of files required for preprocessing.
        _process_hourly_data(start_yyyymm, end_yyyymm, filename_list, explain_dict): Loads and processes data into an hourly format.
        _adjust_and_prepare_data(df_per_hour, start_date, end_optimize_date): Adjusts and prepares the data for further processing.
        _rename_columns(df_per_hour): Renames specific columns in the dataframe based on predefined rules.
        _assign_humidity_and_wet_bulb_temperature(preprocessed_data, df): Assigns humidity data and calculates wet bulb temperature.
        _assign_mean_columns(preprocessed_data, df): Assigns mean columns from the original dataframe to the preprocessed dataframe.
        _assign_system_on_off(preprocessed_data, floors, directions): Assigns system on/off data to the preprocessed dataframe for training.
        _assign_system_on_off_predict(preprocessed_data, floors, directions): Assigns system on/off data to the preprocessed dataframe for prediction.
        _assign_temp_differences(preprocessed_data, df_per_hour, zones): Assigns temperature difference columns to the preprocessed dataframe.
        _assign_operation_modes(preprocessed_data, zones): Assigns operation mode data to the preprocessed dataframe.
        _save_preprocessed_data(preprocessed_data, save_path, overwrite): Saves the preprocessed data to a specified path, merging with existing data if available.
        run(start_date, end_date, end_optimize_date, preprocess_flag): Runs the preprocessing pipeline and merges with existing data if available.
    """

    OFFICE_ZONES = ["4F_西", "4F_東", "5F_西", "5F_東"]

    def __init__(self, combined_data_dict: dict) -> None:
        """Initialize preprocessing with configuration files."""
        # self.df_filename = self._load_preprocessing_config()
        self.combined_data_dict = combined_data_dict

    def _process_hourly_data(
        self,
        start_yyyymm: int,
        end_yyyymm: int,
    ) -> pd.DataFrame:
        """
        Loads and processes data into an hourly format.

        Args:
            start_yyyymm (int): The start year and month in YYYYMM format.
            end_yyyymm (int): The end year and month in YYYYMM format.
            filename_list (list): A list of file names.
            explain_dict (dict): A dictionary containing explain data.

        Returns:
            pd.DataFrame: A dataframe containing hourly data.
        """
        logging.info("Processing data into hourly format...")

        # Create dictionaries to store hourly data
        hourly_data_dict = {}

        # Process each dataframe in the combined data dictionary
        for table_name, df in self.combined_data_dict.items():
            logging.info(f"Processing table: {table_name}")

            # Ensure time column is in datetime format
            if "time" in df.columns:
                # Convert time column to datetime (from format like "20210401000008")
                df["datetime"] = pd.to_datetime(df["time"], format="%Y%m%d%H%M%S")

                # Create searchable hour key for grouping (yyyymmddhh format)
                df["検索用"] = df["datetime"].dt.strftime("%Y%m%d%H")

                # Use 'value' column if it exists
                if "value" in df.columns:
                    # Ensure value column is numeric
                    df["value"] = pd.to_numeric(df["value"], errors="coerce")

                    if (
                        table_name in MIN_MAX_TABLE_NAMES
                    ):  # 1-minute data - calculate max-min difference
                        logging.info(
                            f"{table_name} appears to be 1-minute data, calculating max-min difference"
                        )

                        # Calculate difference between max and min per hour
                        max_values = df.groupby("検索用")["value"].max()
                        min_values = df.groupby("検索用")["value"].min()
                        hourly_values = max_values - min_values

                        # Create DataFrame from the hourly values
                        hourly_df = pd.DataFrame(
                            {
                                "検索用": hourly_values.index,
                                "value": hourly_values.values,
                            }
                        )

                        # Convert 検索用 back to datetime
                        hourly_df["datetime"] = pd.to_datetime(
                            hourly_df["検索用"], format="%Y%m%d%H"
                        )
                    # Determine processing method based on data frequency
                    else:  # 30-minute data - use mean
                        logging.info(
                            f"{table_name} appears to be 30-minute data, calculating mean"
                        )

                        # Resample to hourly frequency and take mean value
                        hourly_df = df.groupby("検索用")["value"].mean().reset_index()

                        # Convert 検索用 back to datetime
                        hourly_df["datetime"] = pd.to_datetime(
                            hourly_df["検索用"], format="%Y%m%d%H"
                        )

                    # Forward fill small gaps (up to 3 hours)
                    hourly_df = hourly_df.sort_values("datetime")
                    hourly_df["value"] = hourly_df["value"].ffill(limit=3)

                    # Add date column for consistency
                    hourly_df["date"] = hourly_df["datetime"].dt.date
                    hourly_df["date"] = pd.to_datetime(hourly_df["date"])

                    # Rename columns to include the table name
                    hourly_df = hourly_df.rename(
                        columns={"value": table_name.replace(".csv", "")}
                    )

                    # Drop the 検索用 column as it's no longer needed
                    if "検索用" in hourly_df.columns:
                        hourly_df = hourly_df.drop(columns=["検索用"])

                    # Store in the hourly data dictionary
                    hourly_data_dict[table_name] = hourly_df

                    logging.info(
                        f"Converted {table_name} to hourly data with shape {hourly_df.shape}"
                    )
                else:
                    logging.warning(
                        f"No 'value' column found in {table_name}, skipping."
                    )
            else:
                logging.warning(f"No 'time' column found in {table_name}, skipping.")

        # Merge all hourly dataframes into one
        if hourly_data_dict:
            # Start with the first dataframe
            merged_df = None

            for table_name, hourly_df in hourly_data_dict.items():
                # For the first dataframe
                if merged_df is None:
                    merged_df = hourly_df.copy()
                    continue

                # Merge on datetime
                merged_df = pd.merge(
                    merged_df, hourly_df, on=["datetime", "date"], how="outer"
                )

            # Check if we have data within the date range
            if merged_df is not None:
                start_date = datetime(
                    int(str(start_yyyymm)[:4]), int(str(start_yyyymm)[4:]), 1
                )
                end_month = int(str(end_yyyymm)[4:])
                end_year = int(str(end_yyyymm)[:4])
                end_date = datetime(end_year, end_month, 1) + pd.offsets.MonthEnd(1)

                # Filter to the date range
                merged_df = merged_df[
                    (merged_df["datetime"] >= start_date)
                    & (merged_df["datetime"] <= end_date)
                ]

                # Sort by datetime
                merged_df = merged_df.sort_values("datetime").reset_index(drop=True)

                # Ensure all value columns are numeric
                value_columns = [
                    col for col in merged_df.columns if col not in ["datetime", "date"]
                ]
                for col in value_columns:
                    merged_df[col] = pd.to_numeric(merged_df[col], errors="coerce")

                logging.info(
                    f"Successfully merged hourly data with shape {merged_df.shape}"
                )
                self.df_per_hour = merged_df
            else:
                logging.error("Failed to merge any dataframes. No data found.")
                self.df_per_hour = pd.DataFrame()
        else:
            logging.error("No hourly data created. Check input dataframes.")
            self.df_per_hour = pd.DataFrame()

    def _rename_columns(self) -> None:
        """
        Renames specific columns in the dataframe based on predefined rules.

        Args:
            df_per_hour (pd.DataFrame): The input dataframe containing hourly data.

        Returns:
            pd.DataFrame: DataFrame with renamed columns.
        """
        logging.info("Renaming columns...")
        # Get the columns that need to be renamed
        columns_to_rename = [
            col for col in self.df_per_hour.columns if col in PREPROCESS_COLUMN_MAPPING
        ]
        # Create the rename dictionary
        rename_dict = {col: PREPROCESS_COLUMN_MAPPING[col] for col in columns_to_rename}
        # Apply the renaming
        self.df_per_hour = self.df_per_hour.rename(columns=rename_dict)

        # Check if our target columns exist after renaming
        target_columns = [
            "室内機消費電力量_kWh_4F_東",
            "室外機消費電力量_kWh_4F_西",
            "室内相対湿度_RH_4F",
            "室内相対湿度_RH_5F",
        ]
        missing_targets = [
            col for col in target_columns if col not in self.df_per_hour.columns
        ]
        if missing_targets:
            logging.error(f"Missing target columns after renaming: {missing_targets}")

    def _adjust_and_prepare_data(
        self,
        start_date: datetime,
        end_optimize_date: datetime,
    ) -> pd.DataFrame:
        """
        Adjusts and prepares the data for further processing.

        Args:
            df_per_hour (pd.DataFrame): The input dataframe containing hourly data.
            start_date (datetime): The start date for data adjustment.
            end_optimize_date (datetime): The end date for data adjustment.

        Returns:
            pd.DataFrame: Adjusted and prepared dataframe.
        """
        logging.info("Adjusting hourly data for the specified period.")
        self.df_per_hour = adjust_hourly_data_for_period(
            self.df_per_hour, start_date, end_optimize_date
        )
        logging.info("Data adjustment completed.")

        self.df_per_hour = self.df_per_hour.reset_index(drop=True)

    def _assign_mean_columns(
        self,
    ) -> None:
        """
        Assigns mean columns from the original dataframe to the preprocessed dataframe.

        Args:
            preprocessed_data (pd.DataFrame): The preprocessed dataframe.
            dataframe (pd.DataFrame): The original dataframe containing the mean columns.

        Returns:
            None
        """
        logging.info("Assigning mean columns...")

        # Define column groups for each area
        column_groups = {
            "室内温度_C_4F執務室_西": [
                col
                for col in self.df_per_hour.columns
                if "4F執務室(西)" in col and "室温モニタ" in col
            ],
            "室内温度_C_4F執務室_東": [
                col
                for col in self.df_per_hour.columns
                if "4F執務室(東)" in col and "室温モニタ" in col
            ],
            "室内温度_C_5F執務室_西": [
                col
                for col in self.df_per_hour.columns
                if "5F執務室(西)" in col and "室温モニタ" in col
            ],
            "室内温度_C_5F執務室_東": [
                col
                for col in self.df_per_hour.columns
                if "5F執務室(東)" in col and "室温モニタ" in col
            ],
            "設定温度_C_4F執務室_西": [
                col
                for col in self.df_per_hour.columns
                if "4F執務室(西)" in col and "設定温度指令ANS" in col
            ],
            "設定温度_C_4F執務室_東": [
                col
                for col in self.df_per_hour.columns
                if "4F執務室(東)" in col and "設定温度指令ANS" in col
            ],
            "設定温度_C_5F執務室_西": [
                col
                for col in self.df_per_hour.columns
                if "5F執務室(西)" in col and "設定温度指令ANS" in col
            ],
            "設定温度_C_5F執務室_東": [
                col
                for col in self.df_per_hour.columns
                if "5F執務室(東)" in col and "設定温度指令ANS" in col
            ],
        }

        for new_col, source_cols in column_groups.items():
            if not source_cols:
                logging.warning(f"No columns found for {new_col}. Skipping.")
                continue

            # Ensure all columns are numeric
            temp_df = self.df_per_hour[source_cols].apply(
                pd.to_numeric, errors="coerce"
            )

            # Calculate mean only for numeric columns
            self.df_per_hour[new_col] = temp_df.mean(axis=1).tolist()

    def _wet_bulb_temp(self) -> None:
        """
        Calculates and assigns the wet bulb temperature to the preprocessed dataframe.

        Args:
            preprocessed_data (pd.DataFrame): DataFrame to store the results.
            dataframe (pd.DataFrame): Source DataFrame containing temperature and humidity.
        """
        logging.info("Calculating wet bulb temperature...")

        # Check if required columns exist
        required_columns = ["外気温度予測値_℃", "外気湿度予測値_RH"]
        missing_columns = [
            col for col in required_columns if col not in self.df_per_hour.columns
        ]

        if missing_columns:
            logging.error(
                f"Missing required columns for wet bulb temperature calculation: {missing_columns}"
            )
            return

        # Calculate and assign wet bulb temperature
        self.df_per_hour["湿球温度_C"] = wetbulb_temp(
            self.df_per_hour[required_columns[0]].tolist(),
            self.df_per_hour[required_columns[1]].tolist(),
        )

        # Verify the column was added
        if "湿球温度_C" in self.df_per_hour.columns:
            logging.info("Successfully added wet bulb temperature column")
        else:
            logging.error("Failed to add wet bulb temperature column")

    def _add_pre_timestep(self) -> None:
        """
        Populates the predict_data DataFrame with previous time step values for specified indoor temperature columns,
        checking first that each column exists.

        Args:
            predict_data (pd.DataFrame): The preprocessed DataFrame for prediction.

        Returns:
            None
        """
        logging.info("Populating previous time step data for predictions...")
        columns = [
            "室内温度_C_4F執務室_西",
            "室内温度_C_4F執務室_東",
            "室内温度_C_5F執務室_西",
            "室内温度_C_5F執務室_東",
        ]
        for column in columns:
            if column in self.df_per_hour.columns:
                self.df_per_hour[column] = add_pre_timestep_data(
                    self.df_per_hour, column
                )
            else:
                logging.warning(
                    f"Column {column} not found in predict_data. Skipping previous time step addition for this column."
                )

    def _assign_system_on_off_predict(self) -> None:
        """
        Assigns system on/off data to the preprocessed dataframe for prediction.

        Args:
            preprocessed_data (pd.DataFrame): The preprocessed dataframe.
            floors (list): A list of floor names.
            directions (list): A list of direction names.

        Returns:
            None
        """
        logging.info("Assigning system on/off data for prediction...")

        floors = ["4", "5"]
        directions = ["西", "東"]

        for floor in floors:
            for direction in directions:
                col_name = f"System_ON_OFF_{floor}F_{direction}"
                self.df_per_hour[col_name] = on_off_judge_for_predict(self.df_per_hour)

    def _assign_temp_differences(self, zones: list) -> None:
        logging.info("Assigning temperature difference columns...")

        # Determine outdoor temperature column
        if "外気温度予測値_℃" in self.df_per_hour.columns:
            outdoor_temp = self.df_per_hour["外気温度予測値_℃"]
        elif "P00049904A_TBL" in self.df_per_hour.columns:
            outdoor_temp = self.df_per_hour["P00049904A_TBL"]
        else:
            logging.error(
                "Outdoor temperature column not found. Skipping temperature difference calculations."
            )
            return

        for zone in zones:
            # Convert zone string to expected label, e.g. "4F_西" -> "4F執務室_西"
            zone_label = zone.replace("_", "執務室_")
            room_temp_col = f"室内温度_C_{zone_label}"
            set_temp_col = f"設定温度_C_{zone_label}"

            # Check if both required columns exist
            if room_temp_col not in self.df_per_hour.columns:
                logging.warning(
                    f"No column found for {room_temp_col}. Skipping temperature differences for zone {zone}."
                )
                continue
            if set_temp_col not in self.df_per_hour.columns:
                logging.warning(
                    f"No column found for {set_temp_col}. Skipping temperature differences for zone {zone}."
                )
                continue

            room_temp = self.df_per_hour[room_temp_col]
            set_temp = self.df_per_hour[set_temp_col]

            self.df_per_hour[f"温度差(外気-室内)_C_{zone}"] = (
                outdoor_temp - room_temp
            ).tolist()
            self.df_per_hour[f"温度差(室内-設定)_C_{zone}"] = (
                room_temp - set_temp
            ).tolist()
            self.df_per_hour[f"温度差(外気-設定)_C_{zone}"] = (
                outdoor_temp - set_temp
            ).tolist()

    def _assign_operation_modes(self, zones: list) -> None:
        """
        Assigns operation mode data to the preprocessed dataframe.

        Args:
            preprocessed_data (pd.DataFrame): The preprocessed dataframe.
            zones (list): A list of zone names.

        Returns:
            None
        """
        logging.info("Assigning operation mode data...")
        for zone in zones:
            self.df_per_hour[f"Operation_Mode_{zone}"] = operation_mode_judge(
                self.df_per_hour
            )

    def save_preprocessed_data(
        self,
        dataframe: pd.DataFrame,
        save_path: str,
        file_name: str = "preprocessed_data",
        overwrite: bool = False,
    ) -> None:
        """
        Saves the preprocessed data to a specified path, merging with existing data if available.

        Args:
            dataframe (pd.DataFrame): The preprocessed dataframe.
            save_path (str): The path to save the preprocessed data.
            overwrite (bool): Flag indicating whether to overwrite existing data.

        Returns:
            None
        """
        if os.path.exists(save_path) and not overwrite:
            logging.info("Merging with existing preprocessed data...")
            existing_data = pd.read_csv(
                save_path + "/" + file_name + ".csv", encoding="cp932"
            )
            if existing_data.index.name is not None:
                existing_data.reset_index(inplace=True)

            dataframe = pd.concat(
                [existing_data, dataframe], ignore_index=True
            ).drop_duplicates(keep="last")
        else:
            logging.info("Overwriting existing preprocessed data...")

        logging.info("Saving preprocessed data...")
        try:
            dataframe.to_csv(
                save_path + "/" + file_name + ".csv", encoding="cp932", index=False
            )
            logging.info("Preprocessed data saved successfully.")
        except Exception as e:
            logging.error(f"Error saving preprocessed data: {e}")

    def run(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """
        Runs the preprocessing pipeline and saves the preprocessed prediction data.

        Args:
            start_date (datetime): The start date of the preprocessing.
            end_date (datetime): The end date of the preprocessing.

        Returns:
            pd.DataFrame: The preprocessed prediction data.
        """

        start_time = time.time()
        start_yyyymm = int(start_date.strftime("%Y%m"))
        end_yyyymm = int(end_date.strftime("%Y%m"))
        logging.info(f"Processing all data from {start_yyyymm} to {end_yyyymm}")
        self.df_per_hour = pd.DataFrame()
        try:
            # Process and adjust the hourly data
            self._process_hourly_data(
                start_yyyymm,
                end_yyyymm,
            )
            self._rename_columns()
            self._adjust_and_prepare_data(start_date, end_date)

            self._assign_mean_columns()
            self._add_pre_timestep()

            self._wet_bulb_temp()
            self._assign_system_on_off_predict()
            self._assign_temp_differences(self.OFFICE_ZONES)
            self._assign_operation_modes(self.OFFICE_ZONES)
            self.df_per_hour["DayType"] = get_weekday(self.df_per_hour)

            # Ensure all required columns exist and are in the correct order
            missing_cols = [
                col
                for col in PREPROCESS_FINAL_COLS
                if col not in self.df_per_hour.columns
            ]
            if missing_cols:
                logging.warning(f"Missing required columns: {missing_cols}")
                # Add missing columns with NaN values
                for col in missing_cols:
                    self.df_per_hour[col] = None
                logging.info("Added missing columns with NaN values")

            # Reorder columns to match PREPROCESS_FINAL_COLS
            self.df_per_hour = self.df_per_hour[PREPROCESS_FINAL_COLS]

            elapsed_time = time.time() - start_time
            logging.info(
                f"Preprocessing completed in {int(elapsed_time // 60)} min {int(elapsed_time % 60)} sec"
            )
            return self.df_per_hour

        except Exception:
            logging.error("An error occurred during preprocessing:")
            logging.error(traceback.format_exc())
            sys.exit(1)


if __name__ == "__main__":
    """
    Function to run the preprocessing pipeline.

    Args:
        start_date (datetime): The start date of the preprocessing.
        end_date (datetime): The end date of the preprocessing.
        end_optimize_date (datetime): The end date for optimization.
        preprocess_flag (int): Indicates whether preprocessing has been done
                    0: Data is available for all 3 years
                    1: There is a period within the specified range that has not been preprocessed
                    2: Data is not available for the full 3-year period

    Returns:
        None
    """

    # Define input parameters
    start_date = datetime(2021, 4, 1)  # Preprocessing start date
    end_date = datetime(2023, 3, 1)  # Preprocessing end date
    end_optimize_date = datetime(2023, 8, 1)  # Optimization end date
    preprocess_flag = 1  # Set the preprocess flag (0, 1, or 2)

    # Define paths for saving preprocessed data
    train_data_path = get_path_from_config_for_outside_base_folder("train_data_path")
    test_data_path = get_path_from_config_for_outside_base_folder("test_data_path")
    predict_data_path = get_path_from_config_for_outside_base_folder(
        "predict_data_path"
    )

    # Initialize and run the preprocessing pipeline
    preprocessor = DataPreprocessor()
    preprocessed_data_list = preprocessor.run(
        start_date,
        end_date,
    )

    logging.info(
        f"Preprocessing completed successfully! and saved to {train_data_path}, {test_data_path}, {predict_data_path}."
    )
