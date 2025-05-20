# prediction_module/wellness_module/comfort_logic.py
import math
import numpy as np
import pandas as pd
from pythermalcomfort.models import pmv
from pythermalcomfort.utilities import v_relative
import logging

logger = logging.getLogger(__name__)

def calculate_comfort(
    df_current_row: pd.DataFrame, # Expects a single-row DataFrame for the current hour
    current_setpoint_column_name: str, 
    master_com_params: pd.DataFrame, # Expects a single-row DataFrame with comfort params
    tdb_previous_hour_setpoint: float # Explicitly pass the dry-bulb temp (setpoint of previous hour)
) -> tuple[float, float, float]: # comfort_index, pmv_value, indoor_relative_humidity
    
    current_pmv_value = np.nan
    comfort_index = np.nan
    # Default indoor_relative_humidity, or it could be an error if master_com_params is misconfigured
    indoor_relative_humidity = 60.0 # A common default, adjust if necessary

    if df_current_row.empty:
        logger.warning("calculate_comfort received an empty df_current_row.")
        return comfort_index, current_pmv_value, indoor_relative_humidity
    if master_com_params.empty:
        logger.warning("calculate_comfort received empty master_com_params.")
        return comfort_index, current_pmv_value, indoor_relative_humidity

    try:
        # master_com_params columns (based on the corrected comfort_parameter_columns):
        # 0: '室内相対湿度 [%] ' (actual name, includes trailing space if read so by pandas)
        # 1: '代謝量 [met]'
        # 2: '着衣量 [clo]'
        # 3: '気流速度 [m/s]'
        
        # Ensure column names used here match EXACTLY what's in master_com_params
        rh_col_name = master_com_params.columns[0] # e.g., '室内相対湿度 [%] '
        met_col_name = master_com_params.columns[1] # e.g., '代謝量 [met]'
        clo_col_name = master_com_params.columns[2] # e.g., '着衣量 [clo]'
        v_col_name = master_com_params.columns[3]   # e.g., '気流速度 [m/s]'

        indoor_relative_humidity = master_com_params[rh_col_name].iloc[0]
        met = master_com_params[met_col_name].iloc[0]
        clo = master_com_params[clo_col_name].iloc[0]
        v = master_com_params[v_col_name].iloc[0]

        vr = v_relative(v=v, met=1.0) # Using met=1.0 for consistency with original finding

        # tr (mean radiant temp) is the current hour's candidate setpoint
        tr_current_hour_setpoint = df_current_row[current_setpoint_column_name].iloc[0]
        
        # tdb (dry bulb temp) is passed as tdb_previous_hour_setpoint

        if pd.notna(tdb_previous_hour_setpoint) and pd.notna(tr_current_hour_setpoint) and \
           pd.notna(vr) and pd.notna(indoor_relative_humidity) and \
           pd.notna(met) and pd.notna(clo):
            
            current_pmv_value = pmv(
                tdb=tdb_previous_hour_setpoint, 
                tr=tr_current_hour_setpoint, 
                vr=vr, 
                rh=indoor_relative_humidity, 
                met=met, 
                clo=clo, 
                standard="ISO"
            )
            if pd.notna(current_pmv_value):
                ppd = 100.0 - 95.0 * math.exp(
                    -0.03353 * pow(current_pmv_value, 4.0) - 0.2179 * pow(current_pmv_value, 2.0)
                )
                comfort_index = 100.0 - ppd
            else: # pmv is nan
                 # logger.debug(f"PMV calculation returned NaN for tdb={tdb_previous_hour_setpoint}, tr={tr_current_hour_setpoint}")
                 comfort_index = 0.0 # Max discomfort
        else:
            # logger.debug(f"One or more inputs to PMV calculation were NaN. tdb_prev={tdb_previous_hour_setpoint}, tr_curr={tr_current_hour_setpoint}")
            comfort_index = 0.0 # Max discomfort if inputs are bad

    except KeyError as e:
        logger.error(f"KeyError in calculate_comfort: {e}. Check column names in master_com_params.")
        logger.error(f"master_com_params columns: {master_com_params.columns.tolist()}")
    except Exception as e:
        logger.error(f"Error in calculate_comfort: {e}", exc_info=True)
        
    return comfort_index, current_pmv_value, indoor_relative_humidity