# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-mouz-vs-vitality-bo3-pYxpz34IEN-t8y4DgB-MSD/mouz-vs-vitality-m3-train.csv`
- round_num: `10`
- rows: `221`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 221 | 1.000 | 0.627457 | 0.652146 | -0.024689 | 57 | 164 | 0.941176 | 0.941176 |
| active/recent utility | 221 | 1.000 | 0.627457 | 0.652146 | -0.024689 | 57 | 164 | 0.941176 | 0.941176 |
| strong utility action | 147 | 0.665 | 0.585606 | 0.615077 | -0.029472 | 33 | 114 | 0.986395 | 0.986395 |
| utility damage | 11 | 0.050 | 0.594462 | 0.534106 | 0.060356 | 11 | 0 | 1.000000 | 1.000000 |
| active smoke/inferno | 139 | 0.629 | 0.584618 | 0.615037 | -0.030419 | 32 | 107 | 0.985612 | 0.985612 |
| recent utility last 5s | 10 | 0.045 | 0.598193 | 0.618358 | -0.020165 | 1 | 9 | 1.000000 | 1.000000 |
| flash effect present | 221 | 1.000 | 0.627457 | 0.652146 | -0.024689 | 57 | 164 | 0.941176 | 0.941176 |

## Active Smoke/Inferno Intervals

- `7.0s` - `69.0s`, rows `125`
- `91.5s` - `98.0s`, rows `14`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `25.5`, LSTM `0.5114`, XGBoost `0.6434`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `25.0`, LSTM `0.5120`, XGBoost `0.6417`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `24.5`, LSTM `0.5145`, XGBoost `0.6416`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `24.0`, LSTM `0.5167`, XGBoost `0.6419`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `26.5`, LSTM `0.5199`, XGBoost `0.6422`, closer `xgboost`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `26.0`, LSTM `0.5214`, XGBoost `0.6434`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `23.5`, LSTM `0.5306`, XGBoost `0.6419`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `20.5`, LSTM `0.5484`, XGBoost `0.6584`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `21.0`, LSTM `0.5441`, XGBoost `0.6527`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `50.0`, LSTM `0.4847`, XGBoost `0.3789`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
