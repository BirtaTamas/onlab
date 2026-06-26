# Local Round Utility State Analysis

- csv_path: `processed_full/blast_open_lisbon/blast-open-lisbon-2025-mouz-vs-falcons-bo3-xBECUqZMcQ8GCwi-GUyz8e/mouz-vs-falcons-m1-train.csv`
- round_num: `14`
- rows: `163`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 163 | 1.000 | 0.858369 | 0.980549 | -0.122180 | 0 | 163 | 1.000000 | 1.000000 |
| active/recent utility | 163 | 1.000 | 0.858369 | 0.980549 | -0.122180 | 0 | 163 | 1.000000 | 1.000000 |
| strong utility action | 97 | 0.595 | 0.837800 | 0.978769 | -0.140970 | 0 | 97 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 97 | 0.595 | 0.837800 | 0.978769 | -0.140970 | 0 | 97 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 163 | 1.000 | 0.858369 | 0.980549 | -0.122180 | 0 | 163 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `11.0s` - `59.0s`, rows `97`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `43.5`, LSTM `0.7785`, XGBoost `0.9766`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `48.0`, LSTM `0.7812`, XGBoost `0.9768`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `43.0`, LSTM `0.7818`, XGBoost `0.9766`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `39.0`, LSTM `0.7852`, XGBoost `0.9766`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `58.5`, LSTM `0.7863`, XGBoost `0.9769`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `52.0`, LSTM `0.7871`, XGBoost `0.9768`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `51.5`, LSTM `0.7884`, XGBoost `0.9768`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `55.5`, LSTM `0.7894`, XGBoost `0.9768`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `42.0`, LSTM `0.7903`, XGBoost `0.9766`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `38.5`, LSTM `0.7905`, XGBoost `0.9766`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
