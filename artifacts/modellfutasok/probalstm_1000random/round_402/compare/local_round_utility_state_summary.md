# Local Round Utility State Analysis

- csv_path: `processed_full/iem_cologne/iem-cologne-2025-vitality-vs-mouz-bo3-kZzxcq2ibUgPOmQh0hZOgn/vitality-vs-mouz-m2-train.csv`
- round_num: `5`
- rows: `176`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 176 | 1.000 | 0.923866 | 0.982177 | -0.058311 | 0 | 176 | 1.000000 | 1.000000 |
| active/recent utility | 176 | 1.000 | 0.923866 | 0.982177 | -0.058311 | 0 | 176 | 1.000000 | 1.000000 |
| strong utility action | 67 | 0.381 | 0.915908 | 0.981384 | -0.065476 | 0 | 67 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 67 | 0.381 | 0.915908 | 0.981384 | -0.065476 | 0 | 67 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 176 | 1.000 | 0.923866 | 0.982177 | -0.058311 | 0 | 176 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `8.5s` - `13.5s`, rows `11`
- `15.5s` - `20.5s`, rows `11`
- `51.5s` - `73.5s`, rows `45`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `54.0`, LSTM `0.8627`, XGBoost `0.9770`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `53.0`, LSTM `0.8687`, XGBoost `0.9772`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `55.5`, LSTM `0.8721`, XGBoost `0.9770`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `61.0`, LSTM `0.8756`, XGBoost `0.9799`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `52.5`, LSTM `0.8738`, XGBoost `0.9772`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `53.5`, LSTM `0.8755`, XGBoost `0.9772`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `62.0`, LSTM `0.8809`, XGBoost `0.9799`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `55.0`, LSTM `0.8778`, XGBoost `0.9760`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `54.5`, LSTM `0.8779`, XGBoost `0.9760`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `60.5`, LSTM `0.8837`, XGBoost `0.9799`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
