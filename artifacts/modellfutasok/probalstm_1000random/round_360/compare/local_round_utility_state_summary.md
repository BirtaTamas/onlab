# Local Round Utility State Analysis

- csv_path: `processed_full/blast_open_london_finals/blast-open-london-2025-finals-faze-vs-g2-bo3-ldI7_iFRuThMOXF8zIbBwX/faze-vs-g2-m1-inferno.csv`
- round_num: `5`
- rows: `247`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 247 | 1.000 | 0.330021 | 0.321029 | 0.008992 | 128 | 119 | 0.453441 | 0.635628 |
| active/recent utility | 247 | 1.000 | 0.330021 | 0.321029 | 0.008992 | 128 | 119 | 0.453441 | 0.635628 |
| strong utility action | 170 | 0.688 | 0.396168 | 0.383665 | 0.012503 | 77 | 93 | 0.347059 | 0.505882 |
| utility damage | 10 | 0.040 | 0.664714 | 0.637982 | 0.026732 | 4 | 6 | 0.000000 | 0.000000 |
| active smoke/inferno | 170 | 0.688 | 0.396168 | 0.383665 | 0.012503 | 77 | 93 | 0.347059 | 0.505882 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 247 | 1.000 | 0.330021 | 0.321029 | 0.008992 | 128 | 119 | 0.453441 | 0.635628 |

## Active Smoke/Inferno Intervals

- `10.0s` - `55.5s`, rows `92`
- `59.0s` - `97.5s`, rows `78`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `68.0`, LSTM `0.5196`, XGBoost `0.3648`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `68.5`, LSTM `0.1984`, XGBoost `0.0810`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `80.0`, LSTM `0.0093`, XGBoost `0.1200`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `56.0`, recent_utility `0`
- seconds `79.5`, LSTM `0.0166`, XGBoost `0.1256`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `56.0`, recent_utility `0`
- seconds `80.5`, LSTM `0.0099`, XGBoost `0.1127`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `56.0`, recent_utility `0`
- seconds `81.0`, LSTM `0.0104`, XGBoost `0.1132`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `56.0`, recent_utility `0`
- seconds `12.5`, LSTM `0.5966`, XGBoost `0.4946`, closer `xgboost`, smoke `3`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `81.5`, LSTM `0.0102`, XGBoost `0.1081`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `56.0`, recent_utility `0`
- seconds `82.0`, LSTM `0.0101`, XGBoost `0.1067`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `56.0`, recent_utility `0`
- seconds `31.0`, LSTM `0.6811`, XGBoost `0.5856`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `36.0`, recent_utility `0`
