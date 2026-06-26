# Local Round Utility State Analysis

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-g2-vs-falcons-bo3-VnJ8NRf6cDNnH9OuqiscGr/g2-vs-falcons-m1-ancient.csv`
- round_num: `16`
- rows: `101`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 101 | 1.000 | 0.535979 | 0.593608 | -0.057629 | 30 | 71 | 0.564356 | 0.841584 |
| active/recent utility | 101 | 1.000 | 0.535979 | 0.593608 | -0.057629 | 30 | 71 | 0.564356 | 0.841584 |
| strong utility action | 88 | 0.871 | 0.529253 | 0.601978 | -0.072724 | 18 | 70 | 0.500000 | 0.909091 |
| utility damage | 11 | 0.109 | 0.522246 | 0.511472 | 0.010774 | 9 | 2 | 1.000000 | 1.000000 |
| active smoke/inferno | 88 | 0.871 | 0.529253 | 0.601978 | -0.072724 | 18 | 70 | 0.500000 | 0.909091 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 101 | 1.000 | 0.535979 | 0.593608 | -0.057629 | 30 | 71 | 0.564356 | 0.841584 |

## Active Smoke/Inferno Intervals

- `6.0s` - `49.5s`, rows `88`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `36.5`, LSTM `0.2273`, XGBoost `0.4919`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `36.0`, LSTM `0.2493`, XGBoost `0.4919`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `37.0`, LSTM `0.2600`, XGBoost `0.4905`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `37.5`, LSTM `0.2623`, XGBoost `0.4877`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `35.5`, LSTM `0.2737`, XGBoost `0.4921`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `38.0`, LSTM `0.5120`, XGBoost `0.7193`, closer `xgboost`, smoke `1`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `41.0`, LSTM `0.6904`, XGBoost `0.8801`, closer `xgboost`, smoke `1`, inferno `2`, utility_damage `12.0`, recent_utility `0`
- seconds `40.5`, LSTM `0.5055`, XGBoost `0.6948`, closer `xgboost`, smoke `1`, inferno `2`, utility_damage `12.0`, recent_utility `0`
- seconds `39.5`, LSTM `0.5072`, XGBoost `0.6897`, closer `xgboost`, smoke `1`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `40.0`, LSTM `0.5123`, XGBoost `0.6897`, closer `xgboost`, smoke `1`, inferno `2`, utility_damage `0.0`, recent_utility `0`
