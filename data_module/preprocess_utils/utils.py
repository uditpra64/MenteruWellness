import logging
from datetime import datetime, timedelta

import jpholiday
import pandas as pd
import psychrolib
from data_module.config.utils import get_path_from_config_for_outside_base_folder
from tqdm import tqdm

"""
Description of this file:
Expected input data: hourly data (provided as separate files for each column)
The script performs preprocessing and generates three types of datasets:
1. Training data
2. Test data
3. Prediction data
Since the calculation methods differ for each dataset type (some parts are shared), each is processed separately and saved.

Shared variable names (partial list):
- df_filename: A table containing information related to preprocessing data
- dict_explain_filename: A dictionary where keys describe the type of preprocessed data, and values are the filenames needed to retrieve that information
- dict_column_name: A dictionary where keys indicate whether the post-processed data is split by system or not (overall), and values are the corresponding names
- dict_3hour_data: A dictionary where keys describe the type of preprocessed data, and values are 3-hour chunks of data (as DataFrames)
- dict_explain_3hdata_per_hour: A dictionary that converts `dict_3hour_data` into hourly-based format
- df_per_hour: A single table that aggregates all data from `dict_explain_3hdata_per_hour`

"""


def get_next_month(current_yyyymm: str) -> str:
    """謖�螳壹＆繧後◆蟷ｴ譛医°繧画ｬ｡縺ｮ譛医ｒ險育ｮ�
    current_yyyymm: str 萓具ｼ�202401
    """

    current_date = datetime.strptime(current_yyyymm, "%Y%m")
    # Add one month
    next_month = current_date.replace(day=28) + timedelta(
        days=4
    )  # 縺薙ｌ縺ｧ蠢�縺壽ｬ｡縺ｮ譛医↓縺ｪ繧�
    next_month = next_month.replace(day=1)  # 1譌･縺ｫ螟画峩

    # Return the next month in 'yyyymm' format
    return next_month.strftime("%Y%m")


def load_and_merge_monthly_data(
    period_folder: str,
    filenames: list[str],
    explain_filename_map: dict[str, str],
    start_yyyymm: str,
    end_yyyymm: str,
) -> dict[str, pd.DataFrame]:
    """
    Loads and merges monthly data within a specified period.

    Parameters:
        period_folder (str): The folder name containing raw data.
        filenames (list[str]): List of filenames (excluding `.csv` extension).
        explain_filename_map (dict[str, str]): A mapping of data descriptions to filenames.
        start_yyyymm (str): The start year-month in 'YYYYMM' format.
        end_yyyymm (str): The end year-month in 'YYYYMM' format.

    Returns:
        dict[str, pd.DataFrame]: A dictionary where keys are data descriptions,
        and values are merged DataFrames for the specified period.
    """
    current_yyyymm = start_yyyymm
    monthly_data = {}  # Dictionary to store data for each month

    # Load data for each month
    while current_yyyymm <= end_yyyymm:
        raw_data_path = get_path_from_config_for_outside_base_folder("rawData_path")
        folder_path = f"{raw_data_path}/{period_folder}/{current_yyyymm}/"

        df_list = []
        for filename in filenames:
            file_path = f"{folder_path}{filename}.csv"
            try:
                df_list.append(pd.read_csv(file_path, index_col=0))
            except FileNotFoundError:
                print(f"Data for period ({current_yyyymm}) is missing: {file_path}")

        if df_list:
            monthly_data[current_yyyymm] = df_list

        current_yyyymm = get_next_month(current_yyyymm)

    # Aggregate data by description (merging different months)
    merged_data = {}
    for index, description in enumerate(explain_filename_map.keys()):
        data_list = [monthly_data[month][index] for month in monthly_data.keys()]
        data_list = [df.astype("float64") for df in data_list]

        # Merge data across months
        merged_df = (
            pd.concat(data_list, ignore_index=True)
            .drop_duplicates(keep="last")
            .reset_index(drop=True)
        )

        # Ensure correct data types
        col1, col2, col3 = merged_df.columns[:3]
        merged_df = merged_df.astype({col1: object, col3: object})
        if merged_df[col2].dtype != float:
            merged_df[col2] = merged_df[col2].astype(float)

        merged_data[description] = merged_df

    return merged_data


def convert_to_3_hour_data(
    data_by_interval: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """
    Converts preprocessed data into an hourly format.

    Parameters:
        data_by_interval (dict[str, pd.DataFrame]): Dictionary where keys are dataset names
                                                    and values are DataFrames with time-series data.

    Returns:
        dict[str, pd.DataFrame]: Dictionary with the same keys but converted to hourly aggregated data.
    """

    hourly_data_dict = data_by_interval.copy()

    for key, df in data_by_interval.items():
        # Extract the hour-based identifier (YYYYMMDDHH) for grouping
        df["hourly_key"] = df.iloc[:, 0].astype(str).apply(lambda x: x[:10])

        if len(df) <= 1_000_000:
            # Data sampled every 30 minutes 竊 compute the mean per hour
            hourly_data_dict[key] = df.groupby("hourly_key").mean(numeric_only=True)
        else:
            # Data sampled every 1 minute 竊 compute accumulated value (max - min) per hour
            hourly_data_dict[key] = (
                df.groupby("hourly_key").max() - df.groupby("hourly_key").min()
            ).iloc[
                :, 1:2
            ]  # Select only relevant columns

    return hourly_data_dict


def merge_hourly_dataframes(
    hourly_data_dict: dict[str, pd.DataFrame], join_type: str = "outer"
) -> pd.DataFrame:
    """
    Merges multiple hourly datasets into a single DataFrame.

    Parameters:
        hourly_data_dict (dict[str, pd.DataFrame]): Dictionary of DataFrames where keys are dataset names.
        join_type (str): Type of join operation - "outer" (default) for full join, "inner" for intersection.

    Returns:
        pd.DataFrame: Merged DataFrame containing all hourly data.
    """

    # Rename columns in each DataFrame to match its corresponding key
    renamed_data_dict = {
        key: df.rename(columns={"value": key}) for key, df in hourly_data_dict.items()
    }

    # Convert dictionary values to a list of DataFrames for merging
    dataframes_list = list(renamed_data_dict.values())

    # Perform merge operation
    merged_df = pd.concat(dataframes_list, axis=1, join=join_type)

    return merged_df


def adjust_hourly_data_for_period(
    df_per_hour_outer: pd.DataFrame, start_study: datetime, end_optimize: datetime
) -> pd.DataFrame:
    """
    Adjusts hourly data for the specified period by calculating average temperatures
    and adding time-related columns.

    Args:
        df_per_hour_outer (pd.DataFrame): The input hourly data
        start_study (datetime): Start date of the study period
        end_optimize (datetime): End date of the optimization period

    Returns:
        pd.DataFrame: Adjusted hourly data with time-related columns
    """
    # Make a copy to avoid modifying the original
    df = df_per_hour_outer.copy()

    # Find temperature columns by name pattern - updated to handle new naming format
    temp_4F_cols = [
        col
        for col in df.columns
        if "4F" in col and ("室温モニタ" in col or "室内温度" in col or "執務室" in col)
    ]
    temp_5F_cols = [
        col
        for col in df.columns
        if "5F" in col and ("室温モニタ" in col or "室内温度" in col or "執務室" in col)
    ]

    if not temp_4F_cols or not temp_5F_cols:
        logging.warning(
            "Could not find temperature columns. Skipping temperature adjustments."
        )
    else:
        # Calculate average temperatures
        temp_4F = df[temp_4F_cols].apply(pd.to_numeric, errors="coerce")
        temp_4F["average"] = temp_4F.mean(axis=1)

        temp_5F = df[temp_5F_cols].apply(pd.to_numeric, errors="coerce")
        temp_5F["average"] = temp_5F.mean(axis=1)

        # Update average temperature columns if they exist
        if "室温(実測平均値)_4F_℃" in df.columns:
            df["室温(実測平均値)_4F_℃"] = temp_4F["average"]
        if "室温(実測平均値)_5F_℃" in df.columns:
            df["室温(実測平均値)_5F_℃"] = temp_5F["average"]

    # Add time-related columns
    df["datetime"] = pd.date_range(start=start_study, end=end_optimize, freq="h")[
        : len(df)
    ]
    df["date"] = df["datetime"].dt.date
    df["fiscal_year"] = df["datetime"].apply(fiscal_year)
    df["month"] = df["datetime"].dt.month
    df["hour"] = df["datetime"].dt.hour
    df["is_holiday"] = df["date"].apply(judge_holiday)

    # Drop columns where all values are NaN (but keep columns with some NaN values)
    df = df.dropna(axis=1, how="all")

    logging.info(f"Adjusted data shape: {df.shape}")
    return df


def fiscal_year(date: datetime.date) -> int:
    """
    莨夊ｨ亥ｹｴ蠎ｦ繧呈歓蜃ｺ (譌･譛ｬ縺ｮ莨夊ｨ亥ｹｴ蠎ｦ縺ｯ4譛医°繧臥ｿ悟ｹｴ縺ｮ3譛�)
    """
    if date.month >= 4:
        return date.year
    else:
        return date.year - 1


def make_df_24h(base_date: datetime.date) -> pd.DataFrame:
    """
    Creates a 24-hour DataFrame for the given date.

    Args:
        base_date (datetime.date): The base date.

    Returns:
        pd.DataFrame: DataFrame with datetime and time-based features for 24 hours.
    """
    # Generate datetime range (hourly for 24h)
    datetime_range = [
        datetime.combine(base_date, datetime.min.time()) + timedelta(hours=i)
        for i in range(24)
    ]

    df = pd.DataFrame({"datetime": datetime_range})
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["date"] = df["datetime"].dt.date
    df["fiscal_year"] = df["datetime"].apply(fiscal_year)
    df["month"] = df["datetime"].dt.month
    df["hour"] = df["datetime"].dt.hour
    df["讀懃ｴ｢逕ｨ"] = df["datetime"].dt.strftime("%Y%m%d%H")

    return df


def judge_holiday(date: datetime.date) -> int:
    """
    Judge whether a given date is a holiday.

    Returns:
        int: 1 if it's a weekend (Saturday/Sunday) or a Japanese public holiday, otherwise 0.
    """
    is_weekend = date.weekday() in [5, 6]  # Saturday or Sunday
    is_national_holiday = jpholiday.is_holiday_name(date) is not None
    return int(is_weekend or is_national_holiday)


def get_weekday(df_per_hour: pd.DataFrame) -> list[str]:
    """
    Returns the weekday name (in Japanese) for each row's date.

    Args:
        df_per_hour (pd.DataFrame): DataFrame with a 'date' column (dtype: datetime).

    Returns:
        list[str]: List of weekday strings in Japanese (e.g., "譛�", "轣ｫ", ...).
    """
    weekday_labels = ["月", "火", "水", "木", "金", "土", "日"]
    return df_per_hour["date"].apply(lambda d: weekday_labels[d.weekday()]).tolist()


def add_pre_timestep_data(data: pd.DataFrame, column: str) -> list:
    """
    Adds the value from the previous timestep for the specified column.

    Args:
        data (pd.DataFrame): Input DataFrame.
        column (str): The name of the column to shift.

    Returns:
        list: List of previous timestep values (with the first value unshifted).
    """
    # Use shift to get the previous row's value; fill the first row with its original value
    shifted = data[column].shift(1)
    shifted.iloc[0] = data[column].iloc[0]
    return shifted.tolist()


def on_off_judge_for_predict(data: pd.DataFrame) -> list[int]:
    """
    ON/OFF判定 for prediction data
    1：ON 0：OFF

    Different from regular on_off_judge:
    - OFF (0) if hour <= 4 or hour >= 22
    - For other hours:
      - OFF (0) if load ratio < 0.2
      - ON (1) if load ratio >= 0.2
    """
    ON_OFF_list = []

    if "空調負荷予測_kWh" not in data.columns:
        logging.error("空調負荷予測_kWh column not found in DataFrame")
        logging.info(f"Available columns: {data.columns.tolist()}")
        return [0] * len(data)

    # Get core hours data for calculating thresholds
    data_a = data[(data["hour"] >= 8) & (data["hour"] <= 18)]
    a = data_a["空調負荷予測_kWh"].copy()
    b = a.describe().tolist()[1] + 2 * a.describe().tolist()[2]  # mean + 2*std
    data_a = data_a[data_a["空調負荷予測_kWh"] <= b]
    max_air_load = data_a["空調負荷予測_kWh"].max()

    if max_air_load == 0:
        logging.error("Maximum air load is 0, which will cause division by zero")
        return [0] * len(data)

    for k in tqdm(range(len(data))):
        # First check time-based conditions
        if data.loc[k, "hour"] <= 4 or data.loc[k, "hour"] >= 22:
            ON_OFF_list.append(0)
        else:
            # For regular hours, check load ratio
            load_ratio = data.loc[k, "空調負荷予測_kWh"] / max_air_load
            if load_ratio < 0.2:
                ON_OFF_list.append(0)
            else:
                ON_OFF_list.append(1)

    return ON_OFF_list


def operation_mode_judge(
    data: pd.DataFrame, temp_column: str = "外気温度予測値_℃"
) -> list[int]:
    """
    Determine the operation mode (cooling or heating) for each row in the input DataFrame.

    The function works on a day-by-day basis. For each unique date:
      - If the month is between June (6) and September (9), the mode is set to cooling (1).
      - If the month is November (11) or later or January through March (<= 3), the mode is set to heating (0).
      - For mid-season months (typically April, May, and October), the mode is determined based on the
        average external temperature during core hours (8:00 to 18:00). If the average temperature is at
        least 20°C, the mode is cooling (1), otherwise heating (0). If there is no core time data, it defaults
        to heating (0).

    Parameters:
      data (pd.DataFrame): DataFrame containing at least the following columns:
      temp_column (str): The name of the column containing the temperature values. Default is "外気温度予測値_℃".

    Returns:
      list[int]: A list of operation modes (1 for cooling, 0 for heating), one value for each row in the input DataFrame.
    """
    result = []

    # Configuration Constants for Readability
    CORE_HOUR_START, CORE_HOUR_END = 8, 18  # Core time range
    COOLING_THRESHOLD_TEMP = (
        20  # Temperature threshold for switching to cooling (in °C)
    )

    # Summer months for cooling : June to September
    SUMMER_MONTH_START, SUMMER_MONTH_END = 6, 9

    # Winter months for heating: November to March
    WINTER_MONTH_START, WINTER_MONTH_END = 11, 3

    # Process data day-by-day by grouping on "date"
    for _, group in data.groupby("date"):
        # Assume that within the same date, the month is constant.
        month = int(group["month"].iloc[0])

        # Determine the operating mode based on the month.
        if SUMMER_MONTH_START <= month <= SUMMER_MONTH_END:
            mode = 1  # Cooling mode for summer months
        elif month >= WINTER_MONTH_START or month <= WINTER_MONTH_END:
            mode = 0  # Heating mode for winter months
        else:
            # For mid-season months (April, May, and October), decide based on average core-hour temperature.
            # Filter the day's data for core hours.
            core_hours_group = group[
                (group["hour"] >= CORE_HOUR_START) & (group["hour"] <= CORE_HOUR_END)
            ]
            if not core_hours_group.empty:
                avg_core_temp = core_hours_group[temp_column].mean()
                mode = 1 if avg_core_temp >= COOLING_THRESHOLD_TEMP else 0
            else:
                mode = 0  # Default to heating if core hours data is missing

        # Extend the result with the determined mode replicated for each record in that day.
        result.extend([mode] * len(group))

    return result


def wetbulb_temp(temp_list: list[float], humid_list: list[float]) -> list[float]:
    """
    Calculate wet-bulb temperature for a list of temperature and humidity values.

    Args:
        temp_list (list[float]): Dry-bulb temperatures in ﾂｰC.
        humid_list (list[float]): Relative humidity values in % (0窶�100).

    Returns:
        list[float]: Wet-bulb temperatures in ﾂｰC.
    """
    pressure = 101325.0  # Standard atmospheric pressure in Pa

    return [
        psychrolib.GetTWetBulbFromRelHum(temp, rh / 100, pressure)
        for temp, rh in zip(temp_list, humid_list)
    ]
