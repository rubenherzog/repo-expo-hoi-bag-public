"""
Shared display labels for exposome variables used in report-facing figures.

Keep all plot scripts on one naming convention so labels do not drift across
figures or supplementary panels.
"""

from __future__ import annotations


EXPOSOME_DISPLAY_NAME: dict[str, str] = {
    "mean_temp_areaw_o": "Mean temp (aw)",
    "mean_temp_o": "Mean temp",
    "maxgtemp_o": "Max temp",
    "mean_prec2_areaw_o": "Precip (aw)",
    "mean_prec2_o": "Precipitation",
    "mean_anomalies_o": "Temp anomalies",
    "mean_anomalies_areaw_o": "Temp anom (aw)",
    "scpdsi_aw_o": "Drought (aw)",
    "scpdsi_o": "Drought",
    "sd_lr_o": "Temp anom SD",
    "PM2.5": "PM2.5",
    "Nitrogen oxide (NOx)": "NOx",
    "Ammonia (NH3) emissions": "NH3",
    "Ammonia (NH\u2083) emissions": "NH\u2083",
    "Carbon monoxide (CO) emissions": "CO",
    "Non-methane volatile organic compounds (NMVOC) emissions": "NMVOC",
    "Sulphur dioxide (SO2) emissions": "SO2",
    "Sulphur dioxide (SO\u2082) emissions": "SO\u2082",
    "Black carbon (BC) emissions": "Black carbon",
    "Average share of green area in city/ urban area": "Green area %",
    "Green area per capita (m2/person)": "Green area/cap",
    "Pop_basic_drinking-water(%)": "Drinking water",
    "Pop_basic_sanitation": "Sanitation",
    "deaths_trans": "Communicable deaths",
    "deaths_notrans": "Non-communicable deaths",
    "Poisoning_mortality_rate": "Poisoning mort.",
    "Electricity_demand (kWh)": "Electricity",
    "agri_emp_o": "Agri employment",
    "climatedisaster_count_o": "Climate disasters",
    "climatedisaster_naffected_o": "Disaster affected",
    "HDI": "HDI",
    "GDP": "GDP",
    "GINI": "GINI",
    "GII": "GII",
    "unemp": "Unemployment",
    "migration": "Migration",
    "rule_law_est": "Rule of law",
    "abs_corrupt_est": "Corruption ctrl",
    "predict_enf_est": "Predict enforce",
    "pers_integ_sec_est": "Personal integr",
    "rights_est": "Rights",
    "access_just_est": "Access justice",
    "free_press_est": "Free press",
    "free_express_est": "Free expression",
    "free_relig_est": "Free religion",
    "representation_est": "Representation",
    "cred_elect_est": "Credible elect",
    "pol_equal_est": "Pol equality",
    "soc_grp_equal_est": "Soc grp equal",
    "econ_equal_est": "Econ equality",
    "elect_part_est": "Elect particip",
    "jud_ind_est": "Judicial indep",
    "basic_welf_est": "Basic welfare",
    "civic_engage_est": "Civic engage",
    "civil_lib_est": "Civil liberties",
    "civil_soc_est": "Civil society",
    "direct_dem_est": "Direct democracy",
    "effect_parl_est": "Effective parl",
    "elected_gov_est": "Elected gov",
    "free_assoc_assem_est": "Free assoc",
    "free_move_est": "Free movement",
    "free_parties_est": "Free parties",
    "gender_equal_est": "Gender equality",
    "inclu_suff_est": "Inclusive suff",
    "local_dem_est": "Local democracy",
    "participation_est": "Participation",
}


def display_label(name: str) -> str:
    key = str(name).strip()
    return EXPOSOME_DISPLAY_NAME.get(key, key)
