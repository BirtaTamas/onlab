# Local Round Utility State Analysis

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-vitality-vs-virtuspro-bo3-8Z0L17IYJlstHvIADVy9G9/vitality-vs-virtus-pro-m3-mirage.csv`
- round_num: `2`
- rows: `204`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 204 | 1.000 | 0.952676 | 0.979707 | -0.027031 | 0 | 204 | 1.000000 | 1.000000 |
| active/recent utility | 204 | 1.000 | 0.952676 | 0.979707 | -0.027031 | 0 | 204 | 1.000000 | 1.000000 |
| strong utility action | 21 | 0.103 | 0.948357 | 0.979098 | -0.030741 | 0 | 21 | 1.000000 | 1.000000 |
| utility damage | 10 | 0.049 | 0.943178 | 0.979606 | -0.036428 | 0 | 10 | 1.000000 | 1.000000 |
| active smoke/inferno | 11 | 0.054 | 0.953065 | 0.978636 | -0.025570 | 0 | 11 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 204 | 1.000 | 0.952676 | 0.979707 | -0.027031 | 0 | 204 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `13.0s` - `18.0s`, rows `11`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `29.0`, LSTM `0.9408`, XGBoost `0.9796`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `61.0`, recent_utility `0`
- seconds `30.5`, LSTM `0.9412`, XGBoost `0.9796`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `61.0`, recent_utility `0`
- seconds `31.5`, LSTM `0.9415`, XGBoost `0.9796`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `61.0`, recent_utility `0`
- seconds `27.0`, LSTM `0.9421`, XGBoost `0.9794`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `61.0`, recent_utility `0`
- seconds `30.0`, LSTM `0.9426`, XGBoost `0.9796`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `61.0`, recent_utility `0`
- seconds `29.5`, LSTM `0.9426`, XGBoost `0.9796`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `61.0`, recent_utility `0`
- seconds `28.5`, LSTM `0.9435`, XGBoost `0.9796`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `61.0`, recent_utility `0`
- seconds `31.0`, LSTM `0.9440`, XGBoost `0.9796`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `61.0`, recent_utility `0`
- seconds `28.0`, LSTM `0.9464`, XGBoost `0.9796`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `61.0`, recent_utility `0`
- seconds `27.5`, LSTM `0.9470`, XGBoost `0.9794`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `61.0`, recent_utility `0`
