# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-chinggis-warriors-vs-lynn-vision-bo3-6KVULP2-Gxo12lI67V9ZfV/chinggis-warriors-vs-lynn-vision-m3-ancient.csv`
- round_num: `7`
- rows: `158`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 158 | 1.000 | 0.721671 | 0.738274 | -0.016604 | 33 | 125 | 1.000000 | 1.000000 |
| active/recent utility | 158 | 1.000 | 0.721671 | 0.738274 | -0.016604 | 33 | 125 | 1.000000 | 1.000000 |
| strong utility action | 117 | 0.741 | 0.684870 | 0.705274 | -0.020403 | 21 | 96 | 1.000000 | 1.000000 |
| utility damage | 11 | 0.070 | 0.595407 | 0.576380 | 0.019026 | 10 | 1 | 1.000000 | 1.000000 |
| active smoke/inferno | 117 | 0.741 | 0.684870 | 0.705274 | -0.020403 | 21 | 96 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 158 | 1.000 | 0.721671 | 0.738274 | -0.016604 | 33 | 125 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `6.0s` - `38.0s`, rows `65`
- `40.5s` - `66.0s`, rows `52`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `42.5`, LSTM `0.5133`, XGBoost `0.5839`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `43.0`, LSTM `0.5141`, XGBoost `0.5839`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `41.5`, LSTM `0.5195`, XGBoost `0.5883`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `42.0`, LSTM `0.5215`, XGBoost `0.5881`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `40.5`, LSTM `0.5237`, XGBoost `0.5883`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `41.0`, LSTM `0.5268`, XGBoost `0.5880`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `43.5`, LSTM `0.5241`, XGBoost `0.5811`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `47.5`, LSTM `0.6861`, XGBoost `0.7401`, closer `xgboost`, smoke `4`, inferno `1`, utility_damage `13.0`, recent_utility `0`
- seconds `37.5`, LSTM `0.5310`, XGBoost `0.5832`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `37.0`, LSTM `0.5331`, XGBoost `0.5815`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
