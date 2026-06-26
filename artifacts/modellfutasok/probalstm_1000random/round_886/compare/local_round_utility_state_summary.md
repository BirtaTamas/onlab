# Local Round Utility State Analysis

- csv_path: `processed_full/blast_rivals_season_1/blast-rivals-2025-season-1-mouz-vs-faze-bo3-q02I_n27c_oaVV09Kplodn/mouz-vs-faze-m2-mirage.csv`
- round_num: `13`
- rows: `206`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 206 | 1.000 | 0.444185 | 0.514986 | -0.070801 | 35 | 171 | 0.330097 | 0.529126 |
| active/recent utility | 206 | 1.000 | 0.444185 | 0.514986 | -0.070801 | 35 | 171 | 0.330097 | 0.529126 |
| strong utility action | 81 | 0.393 | 0.384900 | 0.526704 | -0.141804 | 0 | 81 | 0.135802 | 0.246914 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 81 | 0.393 | 0.384900 | 0.526704 | -0.141804 | 0 | 81 | 0.135802 | 0.246914 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 206 | 1.000 | 0.444185 | 0.514986 | -0.070801 | 35 | 171 | 0.330097 | 0.529126 |

## Active Smoke/Inferno Intervals

- `60.0s` - `81.5s`, rows `44`
- `84.5s` - `102.5s`, rows `37`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `96.0`, LSTM `0.3723`, XGBoost `0.7591`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `99.0`, LSTM `0.5202`, XGBoost `0.8893`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `96.5`, LSTM `0.3912`, XGBoost `0.7591`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `97.0`, LSTM `0.4040`, XGBoost `0.7591`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `98.5`, LSTM `0.4636`, XGBoost `0.8136`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `97.5`, LSTM `0.4103`, XGBoost `0.7591`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `98.0`, LSTM `0.5303`, XGBoost `0.8312`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `101.0`, LSTM `0.6453`, XGBoost `0.9370`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `102.0`, LSTM `0.6702`, XGBoost `0.9368`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `101.5`, LSTM `0.6798`, XGBoost `0.9377`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
