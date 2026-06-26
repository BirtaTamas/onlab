# Local Round Utility State Analysis

- csv_path: `processed_full/iem_cologne/iem-cologne-2025-ninjas-in-pyjamas-vs-natus-vincere-bo3-NyGFAYn592Hu0YfljTVG0N/ninjas-in-pyjamas-vs-natus-vincere-m3-train.csv`
- round_num: `16`
- rows: `154`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 154 | 1.000 | 0.723846 | 0.763999 | -0.040153 | 4 | 150 | 0.928571 | 1.000000 |
| active/recent utility | 154 | 1.000 | 0.723846 | 0.763999 | -0.040153 | 4 | 150 | 0.928571 | 1.000000 |
| strong utility action | 115 | 0.747 | 0.703351 | 0.744282 | -0.040932 | 4 | 111 | 0.956522 | 1.000000 |
| utility damage | 10 | 0.065 | 0.560349 | 0.588776 | -0.028428 | 1 | 9 | 1.000000 | 1.000000 |
| active smoke/inferno | 115 | 0.747 | 0.703351 | 0.744282 | -0.040932 | 4 | 111 | 0.956522 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 154 | 1.000 | 0.723846 | 0.763999 | -0.040153 | 4 | 150 | 0.928571 | 1.000000 |

## Active Smoke/Inferno Intervals

- `7.5s` - `59.0s`, rows `104`
- `66.0s` - `71.0s`, rows `11`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `57.0`, LSTM `0.7563`, XGBoost `0.8755`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `56.5`, LSTM `0.7708`, XGBoost `0.8755`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `55.5`, LSTM `0.7721`, XGBoost `0.8755`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `32.5`, LSTM `0.6570`, XGBoost `0.7578`, closer `xgboost`, smoke `1`, inferno `2`, utility_damage `1.0`, recent_utility `0`
- seconds `55.0`, LSTM `0.7778`, XGBoost `0.8755`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `56.0`, LSTM `0.7781`, XGBoost `0.8755`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `57.5`, LSTM `0.8919`, XGBoost `0.9856`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `54.5`, LSTM `0.7835`, XGBoost `0.8755`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `59.0`, LSTM `0.8405`, XGBoost `0.9264`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `58.0`, LSTM `0.9014`, XGBoost `0.9856`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
