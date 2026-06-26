# Local Round Utility State Analysis

- csv_path: `processed_full/blast_rivals_season_1/blast-rivals-2025-season-1-spirit-vs-flyquest-bo3-fQI-qOiPd1cRkmhkz0Xs5h/spirit-vs-flyquest-m1-mirage.csv`
- round_num: `12`
- rows: `161`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 161 | 1.000 | 0.550184 | 0.569644 | -0.019460 | 68 | 93 | 0.583851 | 0.708075 |
| active/recent utility | 161 | 1.000 | 0.550184 | 0.569644 | -0.019460 | 68 | 93 | 0.583851 | 0.708075 |
| strong utility action | 125 | 0.776 | 0.615128 | 0.619562 | -0.004434 | 53 | 72 | 0.744000 | 0.824000 |
| utility damage | 40 | 0.248 | 0.588876 | 0.576029 | 0.012846 | 20 | 20 | 0.750000 | 0.750000 |
| active smoke/inferno | 112 | 0.696 | 0.612952 | 0.623051 | -0.010099 | 40 | 72 | 0.714286 | 0.803571 |
| recent utility last 5s | 20 | 0.124 | 0.652112 | 0.593970 | 0.058141 | 20 | 0 | 1.000000 | 1.000000 |
| flash effect present | 161 | 1.000 | 0.550184 | 0.569644 | -0.019460 | 68 | 93 | 0.583851 | 0.708075 |

## Active Smoke/Inferno Intervals

- `7.0s` - `53.5s`, rows `94`
- `71.5s` - `80.0s`, rows `18`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `71.5`, LSTM `0.3785`, XGBoost `0.5966`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `72.0`, LSTM `0.3983`, XGBoost `0.5951`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `39.0`, LSTM `0.5133`, XGBoost `0.6703`, closer `xgboost`, smoke `6`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `34.0`, LSTM `0.5175`, XGBoost `0.6645`, closer `xgboost`, smoke `7`, inferno `1`, utility_damage `16.0`, recent_utility `0`
- seconds `35.5`, LSTM `0.4971`, XGBoost `0.6248`, closer `xgboost`, smoke `7`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `72.5`, LSTM `0.4730`, XGBoost `0.5987`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `36.0`, LSTM `0.5004`, XGBoost `0.6241`, closer `xgboost`, smoke `7`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `37.5`, LSTM `0.5012`, XGBoost `0.6241`, closer `xgboost`, smoke `6`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `38.0`, LSTM `0.5048`, XGBoost `0.6256`, closer `xgboost`, smoke `6`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `36.5`, LSTM `0.5043`, XGBoost `0.6241`, closer `xgboost`, smoke `7`, inferno `0`, utility_damage `0.0`, recent_utility `0`
