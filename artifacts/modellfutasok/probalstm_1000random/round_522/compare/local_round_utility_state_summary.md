# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major_stage_2/blasttv-austin-major-2025-stage-2-furia-vs-b8-bo3-3h93b_qbGndTgDFTW66Ud1/furia-vs-b8-m1-mirage.csv`
- round_num: `8`
- rows: `245`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 245 | 1.000 | 0.016418 | 0.045757 | -0.029340 | 231 | 14 | 1.000000 | 1.000000 |
| active/recent utility | 245 | 1.000 | 0.016418 | 0.045757 | -0.029340 | 231 | 14 | 1.000000 | 1.000000 |
| strong utility action | 180 | 0.735 | 0.020916 | 0.056065 | -0.035148 | 180 | 0 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 170 | 0.694 | 0.021566 | 0.056464 | -0.034898 | 170 | 0 | 1.000000 | 1.000000 |
| recent utility last 5s | 10 | 0.041 | 0.009866 | 0.049272 | -0.039406 | 10 | 0 | 1.000000 | 1.000000 |
| flash effect present | 245 | 1.000 | 0.016418 | 0.045757 | -0.029340 | 231 | 14 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `8.5s` - `93.0s`, rows `170`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `25.0`, LSTM `0.0137`, XGBoost `0.0617`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `24.5`, LSTM `0.0138`, XGBoost `0.0617`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `42.5`, LSTM `0.0175`, XGBoost `0.0647`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `77.0`, LSTM `0.0039`, XGBoost `0.0511`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `76.5`, LSTM `0.0040`, XGBoost `0.0511`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `73.0`, LSTM `0.0086`, XGBoost `0.0557`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `76.0`, LSTM `0.0040`, XGBoost `0.0511`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `77.5`, LSTM `0.0041`, XGBoost `0.0511`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `48.0`, LSTM `0.0169`, XGBoost `0.0639`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `41.5`, LSTM `0.0180`, XGBoost `0.0650`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
