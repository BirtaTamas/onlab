# Local Round Utility State Analysis

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-the-mongolz-vs-furia-bo5-6eeTFVdtPEH4qPNc6w4Z3Y/the-mongolz-vs-furia-m5-dust2.csv`
- round_num: `2`
- rows: `136`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 136 | 1.000 | 0.929207 | 0.980073 | -0.050866 | 0 | 136 | 1.000000 | 1.000000 |
| active/recent utility | 136 | 1.000 | 0.929207 | 0.980073 | -0.050866 | 0 | 136 | 1.000000 | 1.000000 |
| strong utility action | 55 | 0.404 | 0.919530 | 0.976730 | -0.057201 | 0 | 55 | 1.000000 | 1.000000 |
| utility damage | 10 | 0.074 | 0.912576 | 0.976165 | -0.063590 | 0 | 10 | 1.000000 | 1.000000 |
| active smoke/inferno | 45 | 0.331 | 0.921075 | 0.976856 | -0.055781 | 0 | 45 | 1.000000 | 1.000000 |
| recent utility last 5s | 10 | 0.074 | 0.922012 | 0.976982 | -0.054970 | 0 | 10 | 1.000000 | 1.000000 |
| flash effect present | 136 | 1.000 | 0.929207 | 0.980073 | -0.050866 | 0 | 136 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `30.0s` - `52.0s`, rows `45`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `9.5`, LSTM `0.8847`, XGBoost `0.9762`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `62.0`, recent_utility `0`
- seconds `10.0`, LSTM `0.8978`, XGBoost `0.9762`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `62.0`, recent_utility `0`
- seconds `10.5`, LSTM `0.8982`, XGBoost `0.9762`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `62.0`, recent_utility `0`
- seconds `40.5`, LSTM `0.9017`, XGBoost `0.9769`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `41.0`, LSTM `0.9045`, XGBoost `0.9767`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `52.0`, LSTM `0.9050`, XGBoost `0.9765`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `11.0`, LSTM `0.9087`, XGBoost `0.9762`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `62.0`, recent_utility `0`
- seconds `39.5`, LSTM `0.9105`, XGBoost `0.9769`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `40.0`, LSTM `0.9111`, XGBoost `0.9769`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `49.5`, LSTM `0.9127`, XGBoost `0.9769`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
