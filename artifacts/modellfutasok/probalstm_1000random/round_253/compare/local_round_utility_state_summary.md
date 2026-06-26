# Local Round Utility State Analysis

- csv_path: `processed_full/iem_cologne/iem-cologne-2025-the-mongolz-vs-natus-vincere-bo3-jwAddb1WR9PRMQexpSMSG8/the-mongolz-vs-natus-vincere-m2-ancient.csv`
- round_num: `6`
- rows: `204`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 204 | 1.000 | 0.834240 | 0.822124 | 0.012116 | 95 | 109 | 1.000000 | 1.000000 |
| active/recent utility | 204 | 1.000 | 0.834240 | 0.822124 | 0.012116 | 95 | 109 | 1.000000 | 1.000000 |
| strong utility action | 112 | 0.549 | 0.759150 | 0.733025 | 0.026125 | 81 | 31 | 1.000000 | 1.000000 |
| utility damage | 22 | 0.108 | 0.887300 | 0.865573 | 0.021727 | 12 | 10 | 1.000000 | 1.000000 |
| active smoke/inferno | 112 | 0.549 | 0.759150 | 0.733025 | 0.026125 | 81 | 31 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 204 | 1.000 | 0.834240 | 0.822124 | 0.012116 | 95 | 109 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `7.0s` - `62.5s`, rows `112`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `38.5`, LSTM `0.9333`, XGBoost `0.8557`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `23.0`, recent_utility `0`
- seconds `32.0`, LSTM `0.5920`, XGBoost `0.5167`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `3.0`, recent_utility `0`
- seconds `39.0`, LSTM `0.9304`, XGBoost `0.8554`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `8.0`, recent_utility `0`
- seconds `32.5`, LSTM `0.5929`, XGBoost `0.5233`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `3.0`, recent_utility `0`
- seconds `10.0`, LSTM `0.5960`, XGBoost `0.5299`, closer `lstm`, smoke `3`, inferno `3`, utility_damage `25.0`, recent_utility `0`
- seconds `9.5`, LSTM `0.5952`, XGBoost `0.5294`, closer `lstm`, smoke `3`, inferno `3`, utility_damage `25.0`, recent_utility `0`
- seconds `33.0`, LSTM `0.5823`, XGBoost `0.5172`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `5.0`, recent_utility `0`
- seconds `26.0`, LSTM `0.5934`, XGBoost `0.5295`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `31.5`, LSTM `0.5809`, XGBoost `0.5172`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `3.0`, recent_utility `0`
- seconds `36.5`, LSTM `0.9070`, XGBoost `0.8440`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `17.0`, recent_utility `0`
