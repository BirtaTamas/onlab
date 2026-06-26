# Local Round Utility State Analysis

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-spirit-vs-heroic-bo3-NSK1XoxyhXK-sIe0204dzp/spirit-vs-heroic-m3-mirage.csv`
- round_num: `9`
- rows: `214`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 214 | 1.000 | 0.476328 | 0.511993 | -0.035665 | 199 | 15 | 0.434579 | 0.144860 |
| active/recent utility | 214 | 1.000 | 0.476328 | 0.511993 | -0.035665 | 199 | 15 | 0.434579 | 0.144860 |
| strong utility action | 190 | 0.888 | 0.508124 | 0.538363 | -0.030239 | 175 | 15 | 0.363158 | 0.094737 |
| utility damage | 10 | 0.047 | 0.729927 | 0.776316 | -0.046389 | 9 | 1 | 0.000000 | 0.000000 |
| active smoke/inferno | 190 | 0.888 | 0.508124 | 0.538363 | -0.030239 | 175 | 15 | 0.363158 | 0.094737 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 214 | 1.000 | 0.476328 | 0.511993 | -0.035665 | 199 | 15 | 0.434579 | 0.144860 |

## Active Smoke/Inferno Intervals

- `5.5s` - `100.0s`, rows `190`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `79.0`, LSTM `0.7103`, XGBoost `0.5798`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `62.5`, LSTM `0.6556`, XGBoost `0.7849`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `63.0`, LSTM `0.6587`, XGBoost `0.7849`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `100.0`, LSTM `0.0172`, XGBoost `0.1321`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `99.5`, LSTM `0.0190`, XGBoost `0.1329`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `79.5`, LSTM `0.6905`, XGBoost `0.5822`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `63.5`, LSTM `0.6812`, XGBoost `0.7849`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `73.0`, LSTM `0.7103`, XGBoost `0.8052`, closer `lstm`, smoke `3`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `72.5`, LSTM `0.7119`, XGBoost `0.8054`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `80.5`, LSTM `0.6711`, XGBoost `0.5824`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
