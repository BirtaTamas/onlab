# Local Round Utility State Analysis

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-flyquest-vs-furia-bo3-kDRQKndVW9qgvAgGZjUFS9/flyquest-vs-furia-m2-dust2.csv`
- round_num: `18`
- rows: `149`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 149 | 1.000 | 0.730423 | 0.766641 | -0.036217 | 18 | 131 | 1.000000 | 1.000000 |
| active/recent utility | 149 | 1.000 | 0.730423 | 0.766641 | -0.036217 | 18 | 131 | 1.000000 | 1.000000 |
| strong utility action | 135 | 0.906 | 0.732648 | 0.772356 | -0.039709 | 10 | 125 | 1.000000 | 1.000000 |
| utility damage | 42 | 0.282 | 0.706748 | 0.738407 | -0.031660 | 4 | 38 | 1.000000 | 1.000000 |
| active smoke/inferno | 135 | 0.906 | 0.732648 | 0.772356 | -0.039709 | 10 | 125 | 1.000000 | 1.000000 |
| recent utility last 5s | 10 | 0.067 | 0.750651 | 0.750865 | -0.000213 | 5 | 5 | 1.000000 | 1.000000 |
| flash effect present | 149 | 1.000 | 0.730423 | 0.766641 | -0.036217 | 18 | 131 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `7.0s` - `74.0s`, rows `135`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `59.5`, LSTM `0.6800`, XGBoost `0.7787`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `72.0`, LSTM `0.7724`, XGBoost `0.8688`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `25.5`, LSTM `0.6908`, XGBoost `0.7854`, closer `xgboost`, smoke `5`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `59.0`, LSTM `0.6903`, XGBoost `0.7787`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `24.0`, LSTM `0.7074`, XGBoost `0.7854`, closer `xgboost`, smoke `5`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `26.5`, LSTM `0.7088`, XGBoost `0.7854`, closer `xgboost`, smoke `5`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `71.5`, LSTM `0.5965`, XGBoost `0.6721`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `48.5`, LSTM `0.7102`, XGBoost `0.7855`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `71.0`, LSTM `0.6131`, XGBoost `0.6882`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `48.0`, LSTM `0.7120`, XGBoost `0.7855`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
