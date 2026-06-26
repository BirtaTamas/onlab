# Local Round Utility State Analysis

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-3dmax-vs-mibr-bo3-O12tFfVag47APQdKBJkGZl/3dmax-vs-mibr-m2-ancient-p3.csv`
- round_num: `8`
- rows: `197`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 197 | 1.000 | 0.104952 | 0.211376 | -0.106423 | 197 | 0 | 1.000000 | 1.000000 |
| active/recent utility | 197 | 1.000 | 0.104952 | 0.211376 | -0.106423 | 197 | 0 | 1.000000 | 1.000000 |
| strong utility action | 141 | 0.716 | 0.123999 | 0.244528 | -0.120529 | 141 | 0 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 141 | 0.716 | 0.123999 | 0.244528 | -0.120529 | 141 | 0 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 197 | 1.000 | 0.104952 | 0.211376 | -0.106423 | 197 | 0 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `8.0s` - `78.0s`, rows `141`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `50.0`, LSTM `0.0209`, XGBoost `0.2929`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `49.5`, LSTM `0.0229`, XGBoost `0.2929`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `48.5`, LSTM `0.0255`, XGBoost `0.2953`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `50.5`, LSTM `0.0237`, XGBoost `0.2929`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `49.0`, LSTM `0.0242`, XGBoost `0.2929`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `52.5`, LSTM `0.0253`, XGBoost `0.2929`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `53.0`, LSTM `0.0254`, XGBoost `0.2929`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `52.0`, LSTM `0.0276`, XGBoost `0.2929`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `51.0`, LSTM `0.0296`, XGBoost `0.2929`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `51.5`, LSTM `0.0310`, XGBoost `0.2929`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
