# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-flyquest-vs-fluxo-ancient-YrTVvYzgDXauKEykMAFJPX/flyquest-vs-fluxo-ancient.csv`
- round_num: `10`
- rows: `170`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 170 | 1.000 | 0.783677 | 0.780752 | 0.002926 | 87 | 83 | 1.000000 | 1.000000 |
| active/recent utility | 170 | 1.000 | 0.783677 | 0.780752 | 0.002926 | 87 | 83 | 1.000000 | 1.000000 |
| strong utility action | 159 | 0.935 | 0.782430 | 0.783730 | -0.001300 | 76 | 83 | 1.000000 | 1.000000 |
| utility damage | 20 | 0.118 | 0.836339 | 0.821160 | 0.015178 | 17 | 3 | 1.000000 | 1.000000 |
| active smoke/inferno | 150 | 0.882 | 0.781323 | 0.785804 | -0.004481 | 67 | 83 | 1.000000 | 1.000000 |
| recent utility last 5s | 20 | 0.118 | 0.772426 | 0.743645 | 0.028781 | 16 | 4 | 1.000000 | 1.000000 |
| flash effect present | 170 | 1.000 | 0.783677 | 0.780752 | 0.002926 | 87 | 83 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `7.0s` - `39.5s`, rows `66`
- `43.0s` - `84.5s`, rows `84`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `46.0`, LSTM `0.5950`, XGBoost `0.7182`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `13.0`, recent_utility `0`
- seconds `46.5`, LSTM `0.6029`, XGBoost `0.7195`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `13.0`, recent_utility `0`
- seconds `45.5`, LSTM `0.6068`, XGBoost `0.7182`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `13.0`, recent_utility `0`
- seconds `47.5`, LSTM `0.6147`, XGBoost `0.7195`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `13.0`, recent_utility `0`
- seconds `45.0`, LSTM `0.6137`, XGBoost `0.7176`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `13.0`, recent_utility `0`
- seconds `47.0`, LSTM `0.6166`, XGBoost `0.7195`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `13.0`, recent_utility `0`
- seconds `50.0`, LSTM `0.6180`, XGBoost `0.7195`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `49.5`, LSTM `0.6185`, XGBoost `0.7198`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `48.0`, LSTM `0.6205`, XGBoost `0.7195`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `13.0`, recent_utility `0`
- seconds `49.0`, LSTM `0.6225`, XGBoost `0.7195`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
