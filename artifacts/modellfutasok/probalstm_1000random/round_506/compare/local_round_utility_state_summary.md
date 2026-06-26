# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-flyquest-vs-tyloo-bo3-b6a1tT091Xo0vOjw70TVd9/flyquest-vs-tyloo-m3-anubis.csv`
- round_num: `2`
- rows: `220`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 220 | 1.000 | 0.915081 | 0.980752 | -0.065671 | 0 | 220 | 1.000000 | 1.000000 |
| active/recent utility | 220 | 1.000 | 0.915081 | 0.980752 | -0.065671 | 0 | 220 | 1.000000 | 1.000000 |
| strong utility action | 146 | 0.664 | 0.902869 | 0.978610 | -0.075741 | 0 | 146 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 146 | 0.664 | 0.902869 | 0.978610 | -0.075741 | 0 | 146 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 220 | 1.000 | 0.915081 | 0.980752 | -0.065671 | 0 | 220 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `12.0s` - `39.0s`, rows `55`
- `45.5s` - `50.5s`, rows `11`
- `54.0s` - `93.5s`, rows `80`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `75.5`, LSTM `0.7967`, XGBoost `0.9754`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `59.0`, LSTM `0.7971`, XGBoost `0.9754`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `57.0`, LSTM `0.7982`, XGBoost `0.9755`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `76.0`, LSTM `0.8000`, XGBoost `0.9754`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `58.0`, LSTM `0.8030`, XGBoost `0.9753`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `58.5`, LSTM `0.8048`, XGBoost `0.9753`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `75.0`, LSTM `0.8058`, XGBoost `0.9758`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `56.5`, LSTM `0.8056`, XGBoost `0.9755`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `56.0`, LSTM `0.8061`, XGBoost `0.9755`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `60.0`, LSTM `0.8073`, XGBoost `0.9755`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
