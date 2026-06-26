# Local Round Utility State Analysis

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-flyquest-vs-furia-bo3-kDRQKndVW9qgvAgGZjUFS9/flyquest-vs-furia-m2-dust2.csv`
- round_num: `20`
- rows: `217`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 217 | 1.000 | 0.450429 | 0.427507 | 0.022921 | 88 | 129 | 0.382488 | 0.350230 |
| active/recent utility | 217 | 1.000 | 0.450429 | 0.427507 | 0.022921 | 88 | 129 | 0.382488 | 0.350230 |
| strong utility action | 199 | 0.917 | 0.439219 | 0.416629 | 0.022590 | 85 | 114 | 0.417085 | 0.381910 |
| utility damage | 20 | 0.092 | 0.592335 | 0.559061 | 0.033274 | 5 | 15 | 0.000000 | 0.000000 |
| active smoke/inferno | 199 | 0.917 | 0.439219 | 0.416629 | 0.022590 | 85 | 114 | 0.417085 | 0.381910 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 217 | 1.000 | 0.450429 | 0.427507 | 0.022921 | 88 | 129 | 0.382488 | 0.350230 |

## Active Smoke/Inferno Intervals

- `9.0s` - `108.0s`, rows `199`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `88.5`, LSTM `0.5232`, XGBoost `0.3114`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `88.0`, LSTM `0.5144`, XGBoost `0.3114`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `68.5`, LSTM `0.5040`, XGBoost `0.3036`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `89.0`, LSTM `0.5068`, XGBoost `0.3135`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `87.5`, LSTM `0.4992`, XGBoost `0.3126`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `87.0`, LSTM `0.4905`, XGBoost `0.3126`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `86.5`, LSTM `0.4814`, XGBoost `0.3126`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `89.5`, LSTM `0.4698`, XGBoost `0.3048`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `94.5`, LSTM `0.3033`, XGBoost `0.1412`, closer `xgboost`, smoke `4`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `86.0`, LSTM `0.4688`, XGBoost `0.3126`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
