# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-legacy-vs-lynn-vision-bo3-80tf5tBYONxHYQuFp0AoSQ/legacy-vs-lynn-vision-m3-nuke.csv`
- round_num: `5`
- rows: `189`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 189 | 1.000 | 0.852290 | 0.877366 | -0.025076 | 26 | 163 | 1.000000 | 1.000000 |
| active/recent utility | 189 | 1.000 | 0.852290 | 0.877366 | -0.025076 | 26 | 163 | 1.000000 | 1.000000 |
| strong utility action | 131 | 0.693 | 0.847579 | 0.876873 | -0.029294 | 11 | 120 | 1.000000 | 1.000000 |
| utility damage | 10 | 0.053 | 0.950899 | 0.990182 | -0.039283 | 0 | 10 | 1.000000 | 1.000000 |
| active smoke/inferno | 131 | 0.693 | 0.847579 | 0.876873 | -0.029294 | 11 | 120 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 189 | 1.000 | 0.852290 | 0.877366 | -0.025076 | 26 | 163 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `7.5s` - `72.5s`, rows `131`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `31.5`, LSTM `0.9239`, XGBoost `0.9864`, closer `xgboost`, smoke `5`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `31.0`, LSTM `0.9270`, XGBoost `0.9864`, closer `xgboost`, smoke `5`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `32.5`, LSTM `0.9320`, XGBoost `0.9870`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `21.0`, LSTM `0.6080`, XGBoost `0.5534`, closer `lstm`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `30.5`, LSTM `0.9322`, XGBoost `0.9864`, closer `xgboost`, smoke `5`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `30.0`, LSTM `0.9334`, XGBoost `0.9857`, closer `xgboost`, smoke `6`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `32.0`, LSTM `0.9349`, XGBoost `0.9864`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `29.5`, LSTM `0.9344`, XGBoost `0.9857`, closer `xgboost`, smoke `6`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `39.0`, LSTM `0.9381`, XGBoost `0.9875`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `42.0`, LSTM `0.9385`, XGBoost `0.9875`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
