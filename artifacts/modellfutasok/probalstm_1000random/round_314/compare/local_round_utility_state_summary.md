# Local Round Utility State Analysis

- csv_path: `processed_full/blast_open_lisbon/blast-open-lisbon-2025-mouz-vs-falcons-bo3-xBECUqZMcQ8GCwi-GUyz8e/mouz-vs-falcons-m1-train.csv`
- round_num: `21`
- rows: `233`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 233 | 1.000 | 0.483721 | 0.602359 | -0.118638 | 15 | 218 | 0.480687 | 0.884120 |
| active/recent utility | 233 | 1.000 | 0.483721 | 0.602359 | -0.118638 | 15 | 218 | 0.480687 | 0.884120 |
| strong utility action | 102 | 0.438 | 0.534630 | 0.637177 | -0.102547 | 2 | 100 | 0.578431 | 0.852941 |
| utility damage | 30 | 0.129 | 0.540312 | 0.627361 | -0.087049 | 0 | 30 | 0.733333 | 1.000000 |
| active smoke/inferno | 102 | 0.438 | 0.534630 | 0.637177 | -0.102547 | 2 | 100 | 0.578431 | 0.852941 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 233 | 1.000 | 0.483721 | 0.602359 | -0.118638 | 15 | 218 | 0.480687 | 0.884120 |

## Active Smoke/Inferno Intervals

- `7.5s` - `35.0s`, rows `56`
- `81.5s` - `88.0s`, rows `14`
- `100.5s` - `116.0s`, rows `32`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `19.5`, LSTM `0.0367`, XGBoost `0.2693`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `115.0`, LSTM `0.4626`, XGBoost `0.6942`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `116.0`, LSTM `0.4736`, XGBoost `0.6928`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `115.5`, LSTM `0.4753`, XGBoost `0.6942`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `86.5`, LSTM `0.2438`, XGBoost `0.4488`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `87.0`, LSTM `0.2463`, XGBoost `0.4488`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `19.0`, LSTM `0.0595`, XGBoost `0.2619`, closer `xgboost`, smoke `4`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `87.5`, LSTM `0.2510`, XGBoost `0.4488`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `88.0`, LSTM `0.2572`, XGBoost `0.4496`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `114.5`, LSTM `0.5179`, XGBoost `0.7021`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
