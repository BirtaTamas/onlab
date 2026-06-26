# Local Round Utility State Analysis

- csv_path: `processed_full/blast_open_london/blast-open-london-2025-spirit-vs-flyquest-bo3-ElcEZT56lTCLJYDcWlMY2d/spirit-vs-flyquest-m1-mirage.csv`
- round_num: `6`
- rows: `230`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 230 | 1.000 | 0.484593 | 0.604521 | -0.119927 | 2 | 228 | 0.265217 | 0.413043 |
| active/recent utility | 230 | 1.000 | 0.484593 | 0.604521 | -0.119927 | 2 | 228 | 0.265217 | 0.413043 |
| strong utility action | 210 | 0.913 | 0.467354 | 0.589469 | -0.122115 | 2 | 208 | 0.223810 | 0.385714 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 202 | 0.878 | 0.470280 | 0.593557 | -0.123277 | 2 | 200 | 0.232673 | 0.400990 |
| recent utility last 5s | 10 | 0.043 | 0.393501 | 0.487994 | -0.094493 | 0 | 10 | 0.000000 | 0.000000 |
| flash effect present | 230 | 1.000 | 0.484593 | 0.604521 | -0.119927 | 2 | 228 | 0.265217 | 0.413043 |

## Active Smoke/Inferno Intervals

- `5.5s` - `31.5s`, rows `53`
- `33.5s` - `88.5s`, rows `111`
- `92.5s` - `99.0s`, rows `14`
- `101.0s` - `112.5s`, rows `24`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `102.0`, LSTM `0.3552`, XGBoost `0.6863`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `3.0`, recent_utility `0`
- seconds `82.5`, LSTM `0.2371`, XGBoost `0.5400`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `82.0`, LSTM `0.2459`, XGBoost `0.5400`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `81.5`, LSTM `0.2491`, XGBoost `0.5400`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `77.5`, LSTM `0.2597`, XGBoost `0.5460`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `81.0`, LSTM `0.2580`, XGBoost `0.5400`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `78.0`, LSTM `0.2647`, XGBoost `0.5455`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `80.0`, LSTM `0.2676`, XGBoost `0.5400`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `83.0`, LSTM `0.2842`, XGBoost `0.5561`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `79.0`, LSTM `0.2715`, XGBoost `0.5400`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
