# Local Round Utility State Analysis

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-3dmax-vs-mibr-bo3-O12tFfVag47APQdKBJkGZl/3dmax-vs-mibr-m2-ancient-p3.csv`
- round_num: `10`
- rows: `157`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 157 | 1.000 | 0.456033 | 0.535297 | -0.079265 | 24 | 133 | 0.356688 | 0.528662 |
| active/recent utility | 157 | 1.000 | 0.456033 | 0.535297 | -0.079265 | 24 | 133 | 0.356688 | 0.528662 |
| strong utility action | 131 | 0.834 | 0.395638 | 0.483828 | -0.088191 | 22 | 109 | 0.274809 | 0.473282 |
| utility damage | 25 | 0.159 | 0.266163 | 0.349358 | -0.083195 | 6 | 19 | 0.080000 | 0.040000 |
| active smoke/inferno | 121 | 0.771 | 0.389904 | 0.481880 | -0.091976 | 22 | 99 | 0.297521 | 0.438017 |
| recent utility last 5s | 10 | 0.064 | 0.465014 | 0.507398 | -0.042384 | 0 | 10 | 0.000000 | 0.900000 |
| flash effect present | 157 | 1.000 | 0.456033 | 0.535297 | -0.079265 | 24 | 133 | 0.356688 | 0.528662 |

## Active Smoke/Inferno Intervals

- `8.0s` - `68.0s`, rows `121`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `58.5`, LSTM `0.1279`, XGBoost `0.5500`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `59.0`, LSTM `0.1504`, XGBoost `0.5487`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `59.5`, LSTM `0.1695`, XGBoost `0.5487`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `60.0`, LSTM `0.2445`, XGBoost `0.5480`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `52.0`, LSTM `0.0160`, XGBoost `0.2606`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `52.5`, LSTM `0.0167`, XGBoost `0.2606`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `51.5`, LSTM `0.0181`, XGBoost `0.2606`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `51.0`, LSTM `0.0185`, XGBoost `0.2606`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `60.5`, LSTM `0.3084`, XGBoost `0.5480`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `50.5`, LSTM `0.0286`, XGBoost `0.2606`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
