# Local Round Utility State Analysis

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-mouz-vs-falcons-bo3-OIe4ELGS25ekkV8Rf6FbR4/mouz-vs-falcons-m3-mirage.csv`
- round_num: `19`
- rows: `153`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 153 | 1.000 | 0.545324 | 0.637635 | -0.092311 | 3 | 150 | 0.274510 | 0.908497 |
| active/recent utility | 153 | 1.000 | 0.545324 | 0.637635 | -0.092311 | 3 | 150 | 0.274510 | 0.908497 |
| strong utility action | 125 | 0.817 | 0.503625 | 0.606881 | -0.103256 | 3 | 122 | 0.208000 | 0.976000 |
| utility damage | 13 | 0.085 | 0.435794 | 0.510872 | -0.075077 | 0 | 13 | 0.000000 | 1.000000 |
| active smoke/inferno | 125 | 0.817 | 0.503625 | 0.606881 | -0.103256 | 3 | 122 | 0.208000 | 0.976000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 153 | 1.000 | 0.545324 | 0.637635 | -0.092311 | 3 | 150 | 0.274510 | 0.908497 |

## Active Smoke/Inferno Intervals

- `6.0s` - `61.0s`, rows `111`
- `63.5s` - `70.0s`, rows `14`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `52.5`, LSTM `0.3286`, XGBoost `0.5378`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `51.5`, LSTM `0.3286`, XGBoost `0.5378`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `52.0`, LSTM `0.3316`, XGBoost `0.5378`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `53.5`, LSTM `0.3347`, XGBoost `0.5378`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `51.0`, LSTM `0.3392`, XGBoost `0.5352`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `53.0`, LSTM `0.3460`, XGBoost `0.5378`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `54.5`, LSTM `0.5142`, XGBoost `0.7031`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `54.0`, LSTM `0.3496`, XGBoost `0.5378`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `50.5`, LSTM `0.3527`, XGBoost `0.5352`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `47.0`, LSTM `0.3609`, XGBoost `0.5333`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
