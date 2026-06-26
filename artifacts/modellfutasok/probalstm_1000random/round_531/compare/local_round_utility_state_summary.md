# Local Round Utility State Analysis

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-flyquest-vs-furia-bo3-kDRQKndVW9qgvAgGZjUFS9/flyquest-vs-furia-m1-inferno.csv`
- round_num: `18`
- rows: `295`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 295 | 1.000 | 0.142466 | 0.165776 | -0.023310 | 247 | 48 | 0.962712 | 1.000000 |
| active/recent utility | 295 | 1.000 | 0.142466 | 0.165776 | -0.023310 | 247 | 48 | 0.962712 | 1.000000 |
| strong utility action | 207 | 0.702 | 0.156807 | 0.186838 | -0.030031 | 177 | 30 | 0.946860 | 1.000000 |
| utility damage | 20 | 0.068 | 0.321602 | 0.356777 | -0.035175 | 14 | 6 | 0.900000 | 1.000000 |
| active smoke/inferno | 207 | 0.702 | 0.156807 | 0.186838 | -0.030031 | 177 | 30 | 0.946860 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 295 | 1.000 | 0.142466 | 0.165776 | -0.023310 | 247 | 48 | 0.962712 | 1.000000 |

## Active Smoke/Inferno Intervals

- `9.5s` - `64.0s`, rows `110`
- `70.5s` - `118.5s`, rows `97`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `86.5`, LSTM `0.3174`, XGBoost `0.1698`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `94.5`, LSTM `0.0343`, XGBoost `0.1595`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `94.0`, LSTM `0.0345`, XGBoost `0.1596`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `95.0`, LSTM `0.0327`, XGBoost `0.1573`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `93.5`, LSTM `0.0369`, XGBoost `0.1601`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `95.5`, LSTM `0.0302`, XGBoost `0.1527`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `93.0`, LSTM `0.0390`, XGBoost `0.1601`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `34.0`, LSTM `0.2048`, XGBoost `0.3209`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `4.0`, recent_utility `0`
- seconds `92.5`, LSTM `0.0462`, XGBoost `0.1601`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `33.5`, LSTM `0.2108`, XGBoost `0.3238`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `11.0`, recent_utility `0`
