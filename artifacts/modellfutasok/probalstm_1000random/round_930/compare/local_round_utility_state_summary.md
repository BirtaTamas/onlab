# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-mouz-vs-vitality-bo3-pYxpz34IEN-t8y4DgB-MSD/mouz-vs-vitality-m2-inferno.csv`
- round_num: `17`
- rows: `180`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 180 | 1.000 | 0.355828 | 0.332458 | 0.023370 | 60 | 120 | 0.527778 | 0.555556 |
| active/recent utility | 180 | 1.000 | 0.355828 | 0.332458 | 0.023370 | 60 | 120 | 0.527778 | 0.555556 |
| strong utility action | 139 | 0.772 | 0.356290 | 0.332346 | 0.023944 | 45 | 94 | 0.539568 | 0.575540 |
| utility damage | 30 | 0.167 | 0.499278 | 0.445349 | 0.053929 | 8 | 22 | 0.366667 | 0.466667 |
| active smoke/inferno | 139 | 0.772 | 0.356290 | 0.332346 | 0.023944 | 45 | 94 | 0.539568 | 0.575540 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 180 | 1.000 | 0.355828 | 0.332458 | 0.023370 | 60 | 120 | 0.527778 | 0.555556 |

## Active Smoke/Inferno Intervals

- `10.5s` - `68.5s`, rows `117`
- `70.0s` - `80.5s`, rows `22`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `53.5`, LSTM `0.4937`, XGBoost `0.2463`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `40.0`, LSTM `0.4347`, XGBoost `0.2353`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `64.0`, recent_utility `0`
- seconds `53.0`, LSTM `0.5191`, XGBoost `0.3462`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `52.5`, LSTM `0.5179`, XGBoost `0.3468`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `52.0`, LSTM `0.5152`, XGBoost `0.3460`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `2.0`, recent_utility `0`
- seconds `51.5`, LSTM `0.5124`, XGBoost `0.3468`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `2.0`, recent_utility `0`
- seconds `51.0`, LSTM `0.5105`, XGBoost `0.3468`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `2.0`, recent_utility `0`
- seconds `54.0`, LSTM `0.2352`, XGBoost `0.0801`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `50.5`, LSTM `0.4954`, XGBoost `0.3454`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `2.0`, recent_utility `0`
- seconds `48.5`, LSTM `0.3878`, XGBoost `0.2402`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `2.0`, recent_utility `0`
