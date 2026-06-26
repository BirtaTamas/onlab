# Local Round Utility State Analysis

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-flyquest-vs-furia-bo3-kDRQKndVW9qgvAgGZjUFS9/flyquest-vs-furia-m2-dust2.csv`
- round_num: `2`
- rows: `187`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 187 | 1.000 | 0.912740 | 0.968761 | -0.056021 | 0 | 187 | 1.000000 | 1.000000 |
| active/recent utility | 187 | 1.000 | 0.912740 | 0.968761 | -0.056021 | 0 | 187 | 1.000000 | 1.000000 |
| strong utility action | 155 | 0.829 | 0.912744 | 0.968692 | -0.055948 | 0 | 155 | 1.000000 | 1.000000 |
| utility damage | 10 | 0.053 | 0.933286 | 0.979818 | -0.046532 | 0 | 10 | 1.000000 | 1.000000 |
| active smoke/inferno | 155 | 0.829 | 0.912744 | 0.968692 | -0.055948 | 0 | 155 | 1.000000 | 1.000000 |
| recent utility last 5s | 10 | 0.053 | 0.980045 | 0.995391 | -0.015346 | 0 | 10 | 1.000000 | 1.000000 |
| flash effect present | 187 | 1.000 | 0.912740 | 0.968761 | -0.056021 | 0 | 187 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `10.5s` - `87.5s`, rows `155`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `67.0`, LSTM `0.6894`, XGBoost `0.9504`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `66.5`, LSTM `0.7316`, XGBoost `0.9504`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `66.0`, LSTM `0.7358`, XGBoost `0.9495`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `73.0`, LSTM `0.5758`, XGBoost `0.7688`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `65.0`, LSTM `0.7626`, XGBoost `0.9495`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `65.5`, LSTM `0.7676`, XGBoost `0.9495`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `64.5`, LSTM `0.7771`, XGBoost `0.9520`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `67.5`, LSTM `0.7918`, XGBoost `0.9509`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `72.0`, LSTM `0.6765`, XGBoost `0.8190`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `72.5`, LSTM `0.6919`, XGBoost `0.8190`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
