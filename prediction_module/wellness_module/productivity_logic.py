# prediction_module/wellness_module/productivity_logic.py
import numpy as np
import pandas as pd
import logging # Optional: for logging warnings/errors

logger = logging.getLogger(__name__)

def calculate_productivity(
    df_current_row: pd.DataFrame, # Expects a single-row DataFrame
    setpoint_column_name: str, 
    master_pro_coeffs: pd.DataFrame, # Expects a single-row DataFrame with coeff columns
    # t parameter is no longer needed if df_current_row is always the single row of interest
) -> float:
    # df_current_row is a single-row DataFrame. Use .iloc[0] to access its values.
    # master_pro_coeffs is also a single-row DataFrame.
    
    current_productivity = np.nan # Default to NaN

    if df_current_row.empty:
        logger.warning("calculate_productivity received an empty DataFrame.")
        return current_productivity
    if master_pro_coeffs.empty:
        logger.warning("calculate_productivity received empty master_pro_coeffs.")
        return current_productivity
    
    try:
        # Access the 'datetime' and setpoint value from the single row using .iloc[0]
        datetime_val = pd.to_datetime(df_current_row["datetime"].iloc[0])
        current_setpoint = df_current_row[setpoint_column_name].iloc[0]
        hour = int(datetime_val.time().strftime("%H"))

        # Access coefficients from the single row of master_pro_coeffs using .iloc[0]
        if 8 <= hour <= 10: # Morning
            current_productivity = (
                master_pro_coeffs["知的生産性_朝_A"].iloc[0] * (current_setpoint ** 2)
                + master_pro_coeffs["知的生産性_朝_B"].iloc[0] * current_setpoint
                + master_pro_coeffs["知的生産性_朝_C"].iloc[0]
            )
        elif 11 <= hour <= 15: # Afternoon (Note: original code had 13-15, check if 11-15 is intended)
                               # Your "2nd row" master data image for "最適化" has coefficients for _昼_
                               # which usually covers the main daytime block.
            current_productivity = (
                master_pro_coeffs["知的生産性_昼_A"].iloc[0] * (current_setpoint ** 2)
                + master_pro_coeffs["知的生産性_昼_B"].iloc[0] * current_setpoint
                + master_pro_coeffs["知的生産性_昼_C"].iloc[0]
            )
        elif 16 <= hour <= 19: # Evening
            current_productivity = (
                master_pro_coeffs["知的生産性_夕_A"].iloc[0] * (current_setpoint ** 2)
                + master_pro_coeffs["知的生産性_夕_B"].iloc[0] * current_setpoint
                + master_pro_coeffs["知的生産性_夕_C"].iloc[0]
            )
        # else: productivity remains np.nan (or define default behavior)

    except KeyError as e:
        logger.error(f"KeyError in calculate_productivity: {e}. Check column names in df_current_row or master_pro_coeffs.")
        logger.error(f"df_current_row columns: {df_current_row.columns.tolist()}")
        logger.error(f"master_pro_coeffs columns: {master_pro_coeffs.columns.tolist()}")
        # current_productivity remains np.nan
    except Exception as e:
        logger.error(f"Error in calculate_productivity: {e}", exc_info=True)
        # current_productivity remains np.nan
        
    return current_productivity