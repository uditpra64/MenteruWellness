# ------------------------------------------------------------
#  Prediction Module (Train and Predict) configs
# ------------------------------------------------------------

# --------------------------------
# 4F_西　Input/Output Features
# --------------------------------
LINEAGE_4F_WEST_INPUT_FEATURES = [
    "外気温度予測値_℃",
    "外気湿度予測値_RH",
    "fiscal_year",
    "month",
    "hour",
    "is_holiday",
    "設定温度_C_4F執務室_西",
    "湿球温度_C",
    "System_ON_OFF_4F_西",
    "温度差(室内-設定)_C_4F_西",
    "温度差(外気-設定)_C_4F_西",
    "Operation_Mode_4F_西",
    "DayType_土",
    "DayType_日",
    "DayType_月",
    "DayType_木",
    "DayType_水",
    "DayType_火",
    "DayType_金",
]
LINEAGE_4F_WEST_OUTPUT_FEATURES = [
    "室内機消費電力量_kWh_4F_西",
    "室外機消費電力量_kWh_4F_西",
]
# --------------------------------
# 4F_東　Input/Output Features
# --------------------------------
LINEAGE_4F_EAST_INPUT_FEATURES = [
    "外気温度予測値_℃",
    "外気湿度予測値_RH",
    "fiscal_year",
    "month",
    "hour",
    "is_holiday",
    "設定温度_C_4F執務室_東",
    "湿球温度_C",
    "System_ON_OFF_4F_東",
    "温度差(室内-設定)_C_4F_東",
    "温度差(外気-設定)_C_4F_東",
    "Operation_Mode_4F_東",
    "DayType_土",
    "DayType_日",
    "DayType_月",
    "DayType_木",
    "DayType_水",
    "DayType_火",
    "DayType_金",
]
LINEAGE_4F_EAST_OUTPUT_FEATURES = [
    "室内機消費電力量_kWh_4F_東",
    "室外機消費電力量_kWh_4F_東",
]

# --------------------------------
# 5F_西　Input/Output Features
# --------------------------------
LINEAGE_5F_WEST_INPUT_FEATURES = [
    "外気温度予測値_℃",
    "外気湿度予測値_RH",
    "fiscal_year",
    "month",
    "hour",
    "is_holiday",
    "設定温度_C_5F執務室_西",
    "湿球温度_C",
    "System_ON_OFF_5F_西",
    "温度差(室内-設定)_C_5F_西",
    "温度差(外気-設定)_C_5F_西",
    "Operation_Mode_5F_西",
    "DayType_土",
    "DayType_日",
    "DayType_月",
    "DayType_木",
    "DayType_水",
    "DayType_火",
    "DayType_金",
]
LINEAGE_5F_WEST_OUTPUT_FEATURES = [
    "室内機消費電力量_kWh_5F_西",
    "室外機消費電力量_kWh_5F_西",
]

# --------------------------------
# 5F_東　Input/Output Features
# --------------------------------
LINEAGE_5F_EAST_INPUT_FEATURES = [
    "外気温度予測値_℃",
    "外気湿度予測値_RH",
    "fiscal_year",
    "month",
    "hour",
    "is_holiday",
    "設定温度_C_5F執務室_東",
    "湿球温度_C",
    "System_ON_OFF_5F_東",
    "温度差(室内-設定)_C_5F_東",
    "温度差(外気-設定)_C_5F_東",
    "Operation_Mode_5F_東",
    "DayType_土",
    "DayType_日",
    "DayType_月",
    "DayType_木",
    "DayType_水",
    "DayType_火",
    "DayType_金",
]
LINEAGE_5F_EAST_OUTPUT_FEATURES = [
    "室内機消費電力量_kWh_5F_東",
    "室外機消費電力量_kWh_5F_東",
]


# ------------------------------------------------------------
# Mapping of Input Features and Output Features by Lineage
# ------------------------------------------------------------
LINEAGES_INPUT_FEATURES_MAPPING = {
    "4F_西": LINEAGE_4F_WEST_INPUT_FEATURES,
    "4F_東": LINEAGE_4F_EAST_INPUT_FEATURES,
    "5F_西": LINEAGE_5F_WEST_INPUT_FEATURES,
    "5F_東": LINEAGE_5F_EAST_INPUT_FEATURES,
}

LINEAGES_OUTPUT_FEATURES_MAPPING = {
    "4F_西": LINEAGE_4F_WEST_OUTPUT_FEATURES,
    "4F_東": LINEAGE_4F_EAST_OUTPUT_FEATURES,
    "5F_西": LINEAGE_5F_WEST_OUTPUT_FEATURES,
    "5F_東": LINEAGE_5F_EAST_OUTPUT_FEATURES,
}

LINEAGES_AGGREGATED_OUTPUT_FEATURE = "総消費電力量_kWh"
