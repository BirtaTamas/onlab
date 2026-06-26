# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-flyquest-vs-tyloo-bo3-b6a1tT091Xo0vOjw70TVd9/flyquest-vs-tyloo-m2-mirage.csv`
- round_num: `14`
- rows: `120`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 120 | 1.000 | 0.954396 | 0.982072 | -0.027676 | 0 | 120 | 1.000000 | 1.000000 |
| active/recent utility | 120 | 1.000 | 0.954396 | 0.982072 | -0.027676 | 0 | 120 | 1.000000 | 1.000000 |
| strong utility action | 65 | 0.542 | 0.953796 | 0.981585 | -0.027789 | 0 | 65 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 55 | 0.458 | 0.955054 | 0.982296 | -0.027242 | 0 | 55 | 1.000000 | 1.000000 |
| recent utility last 5s | 10 | 0.083 | 0.946878 | 0.977676 | -0.030798 | 0 | 10 | 1.000000 | 1.000000 |
| flash effect present | 120 | 1.000 | 0.954396 | 0.982072 | -0.027676 | 0 | 120 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `8.0s` - `13.0s`, rows `11`
- `24.5s` - `46.0s`, rows `44`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `10.5`, LSTM `0.9189`, XGBoost `0.9780`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `12.5`, LSTM `0.9198`, XGBoost `0.9785`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `13.0`, LSTM `0.9204`, XGBoost `0.9785`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `11.0`, LSTM `0.9220`, XGBoost `0.9785`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `11.5`, LSTM `0.9223`, XGBoost `0.9785`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `10.0`, LSTM `0.9226`, XGBoost `0.9779`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `9.5`, LSTM `0.9228`, XGBoost `0.9779`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `9.0`, LSTM `0.9246`, XGBoost `0.9778`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `12.0`, LSTM `0.9256`, XGBoost `0.9785`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `8.5`, LSTM `0.9285`, XGBoost `0.9778`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
