# Local Round Utility State Analysis

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-flyquest-vs-furia-bo3-kDRQKndVW9qgvAgGZjUFS9/flyquest-vs-furia-m1-inferno.csv`
- round_num: `20`
- rows: `114`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 114 | 1.000 | 0.293418 | 0.257535 | 0.035883 | 67 | 47 | 0.938596 | 0.675439 |
| active/recent utility | 114 | 1.000 | 0.293418 | 0.257535 | 0.035883 | 67 | 47 | 0.938596 | 0.675439 |
| strong utility action | 92 | 0.807 | 0.252350 | 0.196292 | 0.056058 | 45 | 47 | 0.923913 | 0.836957 |
| utility damage | 20 | 0.175 | 0.425093 | 0.348292 | 0.076801 | 8 | 12 | 0.850000 | 0.500000 |
| active smoke/inferno | 92 | 0.807 | 0.252350 | 0.196292 | 0.056058 | 45 | 47 | 0.923913 | 0.836957 |
| recent utility last 5s | 10 | 0.088 | 0.391111 | 0.242114 | 0.148997 | 0 | 10 | 1.000000 | 1.000000 |
| flash effect present | 114 | 1.000 | 0.293418 | 0.257535 | 0.035883 | 67 | 47 | 0.938596 | 0.675439 |

## Active Smoke/Inferno Intervals

- `11.0s` - `56.5s`, rows `92`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `38.0`, LSTM `0.3914`, XGBoost `0.1772`, closer `xgboost`, smoke `4`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `39.0`, LSTM `0.3575`, XGBoost `0.1505`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `33.0`, LSTM `0.3881`, XGBoost `0.1870`, closer `xgboost`, smoke `5`, inferno `1`, utility_damage `5.0`, recent_utility `0`
- seconds `38.5`, LSTM `0.3732`, XGBoost `0.1750`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `26.5`, LSTM `0.4673`, XGBoost `0.2752`, closer `xgboost`, smoke `4`, inferno `1`, utility_damage `0.0`, recent_utility `1`
- seconds `37.5`, LSTM `0.3654`, XGBoost `0.1772`, closer `xgboost`, smoke `4`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `32.5`, LSTM `0.3750`, XGBoost `0.1870`, closer `xgboost`, smoke `7`, inferno `0`, utility_damage `5.0`, recent_utility `0`
- seconds `37.0`, LSTM `0.3643`, XGBoost `0.1768`, closer `xgboost`, smoke `4`, inferno `1`, utility_damage `5.0`, recent_utility `0`
- seconds `33.5`, LSTM `0.3736`, XGBoost `0.1870`, closer `xgboost`, smoke `4`, inferno `1`, utility_damage `5.0`, recent_utility `0`
- seconds `34.0`, LSTM `0.3723`, XGBoost `0.1870`, closer `xgboost`, smoke `4`, inferno `1`, utility_damage `5.0`, recent_utility `0`
