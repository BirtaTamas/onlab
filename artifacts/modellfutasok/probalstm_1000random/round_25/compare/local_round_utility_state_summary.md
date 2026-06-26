# Local Round Utility State Analysis

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-furia-vs-lynn-vision-bo3-KVSQ5iZB0TjTG70slfdqOB/furia-vs-lynn-vision-m2-overpass.csv`
- round_num: `4`
- rows: `295`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 295 | 1.000 | 0.084684 | 0.148204 | -0.063520 | 283 | 12 | 0.983051 | 0.911864 |
| active/recent utility | 295 | 1.000 | 0.084684 | 0.148204 | -0.063520 | 283 | 12 | 0.983051 | 0.911864 |
| strong utility action | 216 | 0.732 | 0.067184 | 0.139582 | -0.072398 | 216 | 0 | 1.000000 | 0.962963 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 216 | 0.732 | 0.067184 | 0.139582 | -0.072398 | 216 | 0 | 1.000000 | 0.962963 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 295 | 1.000 | 0.084684 | 0.148204 | -0.063520 | 283 | 12 | 0.983051 | 0.911864 |

## Active Smoke/Inferno Intervals

- `9.0s` - `33.0s`, rows `49`
- `41.0s` - `124.0s`, rows `167`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `47.0`, LSTM `0.0822`, XGBoost `0.3106`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `47.5`, LSTM `0.0819`, XGBoost `0.3102`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `48.5`, LSTM `0.0812`, XGBoost `0.3086`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `48.0`, LSTM `0.0832`, XGBoost `0.3086`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `46.5`, LSTM `0.0884`, XGBoost `0.3106`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `45.5`, LSTM `0.0960`, XGBoost `0.3182`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `45.0`, LSTM `0.0841`, XGBoost `0.3018`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `46.0`, LSTM `0.0970`, XGBoost `0.3112`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `33.0`, LSTM `0.0864`, XGBoost `0.2960`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `20.5`, LSTM `0.0879`, XGBoost `0.2962`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
