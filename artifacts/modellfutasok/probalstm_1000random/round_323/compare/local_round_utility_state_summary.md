# Local Round Utility State Analysis

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-vitality-vs-astralis-bo3-Zley6FZuKcttfrliAqsvWJ/astralis-vs-vitality-m1-inferno.csv`
- round_num: `16`
- rows: `199`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 199 | 1.000 | 0.688879 | 0.709510 | -0.020631 | 64 | 135 | 0.894472 | 1.000000 |
| active/recent utility | 199 | 1.000 | 0.688879 | 0.709510 | -0.020631 | 64 | 135 | 0.894472 | 1.000000 |
| strong utility action | 171 | 0.859 | 0.707248 | 0.726888 | -0.019640 | 64 | 107 | 1.000000 | 1.000000 |
| utility damage | 22 | 0.111 | 0.594801 | 0.676951 | -0.082150 | 0 | 22 | 1.000000 | 1.000000 |
| active smoke/inferno | 170 | 0.854 | 0.708107 | 0.727498 | -0.019391 | 64 | 106 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 199 | 1.000 | 0.688879 | 0.709510 | -0.020631 | 64 | 135 | 0.894472 | 1.000000 |

## Active Smoke/Inferno Intervals

- `11.5s` - `54.5s`, rows `87`
- `55.5s` - `96.5s`, rows `83`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `36.0`, LSTM `0.5271`, XGBoost `0.6963`, closer `xgboost`, smoke `3`, inferno `2`, utility_damage `32.0`, recent_utility `0`
- seconds `35.5`, LSTM `0.5336`, XGBoost `0.6914`, closer `xgboost`, smoke `3`, inferno `2`, utility_damage `32.0`, recent_utility `0`
- seconds `37.0`, LSTM `0.5406`, XGBoost `0.6977`, closer `xgboost`, smoke `3`, inferno `2`, utility_damage `29.0`, recent_utility `0`
- seconds `36.5`, LSTM `0.5509`, XGBoost `0.6988`, closer `xgboost`, smoke `3`, inferno `2`, utility_damage `32.0`, recent_utility `0`
- seconds `38.0`, LSTM `0.5493`, XGBoost `0.6967`, closer `xgboost`, smoke `3`, inferno `2`, utility_damage `7.0`, recent_utility `0`
- seconds `20.0`, LSTM `0.5616`, XGBoost `0.7088`, closer `xgboost`, smoke `2`, inferno `2`, utility_damage `44.0`, recent_utility `0`
- seconds `18.0`, LSTM `0.5655`, XGBoost `0.7088`, closer `xgboost`, smoke `2`, inferno `3`, utility_damage `44.0`, recent_utility `0`
- seconds `19.5`, LSTM `0.5655`, XGBoost `0.7088`, closer `xgboost`, smoke `2`, inferno `2`, utility_damage `44.0`, recent_utility `0`
- seconds `35.0`, LSTM `0.5508`, XGBoost `0.6921`, closer `xgboost`, smoke `3`, inferno `2`, utility_damage `32.0`, recent_utility `0`
- seconds `37.5`, LSTM `0.5561`, XGBoost `0.6967`, closer `xgboost`, smoke `3`, inferno `2`, utility_damage `22.0`, recent_utility `0`
