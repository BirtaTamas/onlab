# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major_stage_2/blasttv-austin-major-2025-stage-2-faze-vs-heroic-dust2-PtQF8ASKD1754yZQHk6148/faze-vs-heroic-dust2.csv`
- round_num: `12`
- rows: `108`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 108 | 1.000 | 0.805198 | 0.803256 | 0.001943 | 43 | 65 | 1.000000 | 1.000000 |
| active/recent utility | 108 | 1.000 | 0.805198 | 0.803256 | 0.001943 | 43 | 65 | 1.000000 | 1.000000 |
| strong utility action | 91 | 0.843 | 0.834018 | 0.840783 | -0.006766 | 27 | 64 | 1.000000 | 1.000000 |
| utility damage | 10 | 0.093 | 0.979058 | 0.995626 | -0.016568 | 0 | 10 | 1.000000 | 1.000000 |
| active smoke/inferno | 91 | 0.843 | 0.834018 | 0.840783 | -0.006766 | 27 | 64 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 108 | 1.000 | 0.805198 | 0.803256 | 0.001943 | 43 | 65 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `8.0s` - `53.0s`, rows `91`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `21.0`, LSTM `0.7007`, XGBoost `0.7550`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `11.0`, LSTM `0.6090`, XGBoost `0.5719`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `28.0`, LSTM `0.8717`, XGBoost `0.9067`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `26.0`, LSTM `0.8742`, XGBoost `0.9085`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `27.5`, LSTM `0.8782`, XGBoost `0.9076`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `8.0`, LSTM `0.5970`, XGBoost `0.5678`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `26.5`, LSTM `0.8825`, XGBoost `0.9076`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `23.0`, LSTM `0.8848`, XGBoost `0.9073`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `27.0`, LSTM `0.8851`, XGBoost `0.9076`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `24.0`, LSTM `0.8859`, XGBoost `0.9078`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
