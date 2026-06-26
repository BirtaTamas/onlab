# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-furia-vs-virtuspro-bo3-E_bOFuD3YUjLJCO2xRj0mq/furia-vs-virtus-pro-m1-mirage.csv`
- round_num: `4`
- rows: `173`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 173 | 1.000 | 0.949393 | 0.978779 | -0.029386 | 0 | 173 | 1.000000 | 1.000000 |
| active/recent utility | 173 | 1.000 | 0.949393 | 0.978779 | -0.029386 | 0 | 173 | 1.000000 | 1.000000 |
| strong utility action | 126 | 0.728 | 0.948767 | 0.978125 | -0.029358 | 0 | 126 | 1.000000 | 1.000000 |
| utility damage | 20 | 0.116 | 0.956832 | 0.978218 | -0.021386 | 0 | 20 | 1.000000 | 1.000000 |
| active smoke/inferno | 121 | 0.699 | 0.948158 | 0.978140 | -0.029982 | 0 | 121 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 173 | 1.000 | 0.949393 | 0.978779 | -0.029386 | 0 | 173 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `7.0s` - `39.5s`, rows `66`
- `42.5s` - `47.5s`, rows `11`
- `50.0s` - `71.5s`, rows `44`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `7.5`, LSTM `0.9021`, XGBoost `0.9738`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `8.0`, LSTM `0.9107`, XGBoost `0.9740`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `7.0`, LSTM `0.9141`, XGBoost `0.9738`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `8.5`, LSTM `0.9156`, XGBoost `0.9750`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `9.5`, LSTM `0.9176`, XGBoost `0.9750`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `10.0`, LSTM `0.9187`, XGBoost `0.9751`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `9.0`, LSTM `0.9194`, XGBoost `0.9750`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `10.5`, LSTM `0.9266`, XGBoost `0.9752`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `11.5`, LSTM `0.9330`, XGBoost `0.9776`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `11.0`, LSTM `0.9333`, XGBoost `0.9776`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
