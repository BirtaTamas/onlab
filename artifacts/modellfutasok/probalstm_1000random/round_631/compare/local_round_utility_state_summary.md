# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-furia-vs-virtuspro-bo3-E_bOFuD3YUjLJCO2xRj0mq/furia-vs-virtus-pro-m1-mirage.csv`
- round_num: `2`
- rows: `230`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 230 | 1.000 | 0.610388 | 0.676314 | -0.065925 | 14 | 216 | 0.469565 | 0.626087 |
| active/recent utility | 230 | 1.000 | 0.610388 | 0.676314 | -0.065925 | 14 | 216 | 0.469565 | 0.626087 |
| strong utility action | 146 | 0.635 | 0.476625 | 0.562339 | -0.085713 | 14 | 132 | 0.267123 | 0.513699 |
| utility damage | 11 | 0.048 | 0.417060 | 0.504097 | -0.087037 | 1 | 10 | 0.363636 | 0.727273 |
| active smoke/inferno | 136 | 0.591 | 0.493457 | 0.581486 | -0.088029 | 12 | 124 | 0.286765 | 0.551471 |
| recent utility last 5s | 10 | 0.043 | 0.247714 | 0.301938 | -0.054224 | 2 | 8 | 0.000000 | 0.000000 |
| flash effect present | 230 | 1.000 | 0.610388 | 0.676314 | -0.065925 | 14 | 216 | 0.469565 | 0.626087 |

## Active Smoke/Inferno Intervals

- `8.5s` - `30.0s`, rows `44`
- `34.5s` - `80.0s`, rows `92`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `38.5`, LSTM `0.1611`, XGBoost `0.3876`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `50.0`, LSTM `0.3624`, XGBoost `0.5853`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `49.5`, LSTM `0.3688`, XGBoost `0.5853`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `49.0`, LSTM `0.3751`, XGBoost `0.5853`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `48.0`, LSTM `0.3742`, XGBoost `0.5834`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `51.0`, LSTM `0.3878`, XGBoost `0.5853`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `48.5`, LSTM `0.3879`, XGBoost `0.5839`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `50.5`, LSTM `0.3929`, XGBoost `0.5853`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `55.0`, LSTM `0.4081`, XGBoost `0.5931`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `51.5`, LSTM `0.4011`, XGBoost `0.5853`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
