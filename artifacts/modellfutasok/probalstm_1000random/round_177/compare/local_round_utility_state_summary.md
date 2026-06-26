# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-betboom-vs-nemiga-train-khA7BVyAiKBjWcyTrFzube/betboom-vs-nemiga-train.csv`
- round_num: `6`
- rows: `182`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 182 | 1.000 | 0.596802 | 0.635166 | -0.038364 | 32 | 150 | 0.978022 | 0.983516 |
| active/recent utility | 182 | 1.000 | 0.596802 | 0.635166 | -0.038364 | 32 | 150 | 0.978022 | 0.983516 |
| strong utility action | 153 | 0.841 | 0.596048 | 0.629015 | -0.032967 | 27 | 126 | 0.980392 | 0.980392 |
| utility damage | 10 | 0.055 | 0.561146 | 0.545429 | 0.015717 | 10 | 0 | 1.000000 | 1.000000 |
| active smoke/inferno | 153 | 0.841 | 0.596048 | 0.629015 | -0.032967 | 27 | 126 | 0.980392 | 0.980392 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 182 | 1.000 | 0.596802 | 0.635166 | -0.038364 | 32 | 150 | 0.978022 | 0.983516 |

## Active Smoke/Inferno Intervals

- `7.5s` - `46.0s`, rows `78`
- `47.5s` - `84.5s`, rows `75`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `76.0`, LSTM `0.7031`, XGBoost `0.8755`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `73.0`, LSTM `0.6999`, XGBoost `0.8716`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `79.5`, LSTM `0.7074`, XGBoost `0.8760`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `77.5`, LSTM `0.7081`, XGBoost `0.8767`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `78.5`, LSTM `0.7116`, XGBoost `0.8767`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `79.0`, LSTM `0.7131`, XGBoost `0.8767`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `78.0`, LSTM `0.7153`, XGBoost `0.8767`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `80.0`, LSTM `0.7192`, XGBoost `0.8767`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `76.5`, LSTM `0.7201`, XGBoost `0.8766`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `74.0`, LSTM `0.7114`, XGBoost `0.8664`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
