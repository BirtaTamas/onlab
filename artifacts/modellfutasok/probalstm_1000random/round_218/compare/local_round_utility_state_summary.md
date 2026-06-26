# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major_stage_2/blasttv-austin-major-2025-stage-2-og-vs-falcons-bo3-Q3yO3LacAwamKdCbguw7-l/og-vs-falcons-m1-dust2.csv`
- round_num: `1`
- rows: `150`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 150 | 1.000 | 0.605833 | 0.633365 | -0.027532 | 88 | 62 | 0.900000 | 0.326667 |
| active/recent utility | 150 | 1.000 | 0.605833 | 0.633365 | -0.027532 | 88 | 62 | 0.900000 | 0.326667 |
| strong utility action | 96 | 0.640 | 0.647115 | 0.691841 | -0.044726 | 44 | 52 | 0.895833 | 0.447917 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 96 | 0.640 | 0.647115 | 0.691841 | -0.044726 | 44 | 52 | 0.895833 | 0.447917 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 150 | 1.000 | 0.605833 | 0.633365 | -0.027532 | 88 | 62 | 0.900000 | 0.326667 |

## Active Smoke/Inferno Intervals

- `18.5s` - `40.0s`, rows `44`
- `46.5s` - `72.0s`, rows `52`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `51.0`, LSTM `0.4245`, XGBoost `0.6764`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `50.5`, LSTM `0.2073`, XGBoost `0.4551`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `52.5`, LSTM `0.6615`, XGBoost `0.8703`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `54.0`, LSTM `0.6896`, XGBoost `0.8843`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `54.5`, LSTM `0.6844`, XGBoost `0.8745`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `53.0`, LSTM `0.6932`, XGBoost `0.8703`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `53.5`, LSTM `0.7016`, XGBoost `0.8754`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `47.5`, LSTM `0.3139`, XGBoost `0.4819`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `66.5`, LSTM `0.7993`, XGBoost `0.9643`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `55.0`, LSTM `0.7248`, XGBoost `0.8840`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
