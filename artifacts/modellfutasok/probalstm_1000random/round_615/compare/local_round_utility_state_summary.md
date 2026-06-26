# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-virtuspro-vs-legacy-ancient-7ivruObh5LTTVaCYe9h-YO/virtus-pro-vs-legacy-ancient.csv`
- round_num: `11`
- rows: `196`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 196 | 1.000 | 0.571224 | 0.668107 | -0.096883 | 7 | 189 | 0.627551 | 0.806122 |
| active/recent utility | 196 | 1.000 | 0.571224 | 0.668107 | -0.096883 | 7 | 189 | 0.627551 | 0.806122 |
| strong utility action | 122 | 0.622 | 0.476772 | 0.593483 | -0.116711 | 2 | 120 | 0.467213 | 0.729508 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 114 | 0.582 | 0.482726 | 0.599787 | -0.117061 | 2 | 112 | 0.500000 | 0.736842 |
| recent utility last 5s | 22 | 0.112 | 0.393267 | 0.501538 | -0.108271 | 0 | 22 | 0.000000 | 0.545455 |
| flash effect present | 196 | 1.000 | 0.571224 | 0.668107 | -0.096883 | 7 | 189 | 0.627551 | 0.806122 |

## Active Smoke/Inferno Intervals

- `6.5s` - `56.0s`, rows `100`
- `71.5s` - `78.0s`, rows `14`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `6.5`, LSTM `0.3054`, XGBoost `0.5074`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `2`
- seconds `6.0`, LSTM `0.3106`, XGBoost `0.5079`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `2`
- seconds `77.5`, LSTM `0.3999`, XGBoost `0.5827`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `45.0`, LSTM `0.5310`, XGBoost `0.7133`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `44.5`, LSTM `0.5327`, XGBoost `0.7133`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `33.5`, LSTM `0.5511`, XGBoost `0.7305`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `30.5`, LSTM `0.5537`, XGBoost `0.7301`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `35.5`, LSTM `0.5503`, XGBoost `0.7258`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `45.5`, LSTM `0.5385`, XGBoost `0.7133`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `46.0`, LSTM `0.5387`, XGBoost `0.7133`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
