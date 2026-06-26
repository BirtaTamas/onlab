# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major_stage_2/blasttv-austin-major-2025-stage-2-b8-vs-lynn-vision-bo3-Whl3pjYuIoHffY1VOn8vws/b8-vs-lynn-vision-m1-dust2.csv`
- round_num: `3`
- rows: `196`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 196 | 1.000 | 0.183446 | 0.292856 | -0.109411 | 196 | 0 | 0.974490 | 0.816327 |
| active/recent utility | 196 | 1.000 | 0.183446 | 0.292856 | -0.109411 | 196 | 0 | 0.974490 | 0.816327 |
| strong utility action | 160 | 0.816 | 0.182426 | 0.286943 | -0.104516 | 160 | 0 | 0.968750 | 0.843750 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 160 | 0.816 | 0.182426 | 0.286943 | -0.104516 | 160 | 0 | 0.968750 | 0.843750 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 196 | 1.000 | 0.183446 | 0.292856 | -0.109411 | 196 | 0 | 0.974490 | 0.816327 |

## Active Smoke/Inferno Intervals

- `7.5s` - `45.0s`, rows `76`
- `49.0s` - `55.5s`, rows `14`
- `57.5s` - `92.0s`, rows `70`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `86.5`, LSTM `0.5578`, XGBoost `0.8719`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `90.5`, LSTM `0.2976`, XGBoost `0.5434`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `86.0`, LSTM `0.4077`, XGBoost `0.6486`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `90.0`, LSTM `0.3325`, XGBoost `0.5711`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `70.5`, LSTM `0.2083`, XGBoost `0.4429`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `89.5`, LSTM `0.3302`, XGBoost `0.5644`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `92.0`, LSTM `0.3474`, XGBoost `0.5772`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `91.5`, LSTM `0.3497`, XGBoost `0.5772`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `87.0`, LSTM `0.6492`, XGBoost `0.8741`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `70.0`, LSTM `0.2394`, XGBoost `0.4488`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
