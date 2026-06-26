# Local Round Utility State Analysis

- csv_path: `processed_full/blast_open_london/blast-open-london-2025-g2-vs-liquid-bo3-w6HylYj4nF7GNnrWujmZUZ/g2-vs-liquid-m2-inferno.csv`
- round_num: `14`
- rows: `142`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 142 | 1.000 | 0.927832 | 0.980432 | -0.052600 | 0 | 142 | 1.000000 | 1.000000 |
| active/recent utility | 142 | 1.000 | 0.927832 | 0.980432 | -0.052600 | 0 | 142 | 1.000000 | 1.000000 |
| strong utility action | 107 | 0.754 | 0.927635 | 0.980438 | -0.052803 | 0 | 107 | 1.000000 | 1.000000 |
| utility damage | 20 | 0.141 | 0.926769 | 0.979323 | -0.052554 | 0 | 20 | 1.000000 | 1.000000 |
| active smoke/inferno | 107 | 0.754 | 0.927635 | 0.980438 | -0.052803 | 0 | 107 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 142 | 1.000 | 0.927832 | 0.980432 | -0.052600 | 0 | 142 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `11.5s` - `61.5s`, rows `101`
- `68.0s` - `70.5s`, rows `6`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `44.5`, LSTM `0.8829`, XGBoost `0.9801`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `44.0`, LSTM `0.8882`, XGBoost `0.9801`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `34.5`, LSTM `0.8886`, XGBoost `0.9788`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `32.0`, LSTM `0.8941`, XGBoost `0.9788`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `32.5`, LSTM `0.8940`, XGBoost `0.9787`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `42.5`, LSTM `0.8963`, XGBoost `0.9789`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `35.0`, LSTM `0.8967`, XGBoost `0.9788`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `43.5`, LSTM `0.8986`, XGBoost `0.9801`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `45.0`, LSTM `0.8993`, XGBoost `0.9800`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `40.5`, LSTM `0.8996`, XGBoost `0.9785`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
