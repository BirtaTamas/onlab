# Local Round Utility State Analysis

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-lynn-vision-vs-vitality-bo3-ZpOL0o26IrRvvgFRbFxVou/lynn-vision-vs-vitality-m1-dust2.csv`
- round_num: `15`
- rows: `261`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 261 | 1.000 | 0.295235 | 0.353387 | -0.058152 | 207 | 54 | 0.946360 | 1.000000 |
| active/recent utility | 261 | 1.000 | 0.295235 | 0.353387 | -0.058152 | 207 | 54 | 0.946360 | 1.000000 |
| strong utility action | 200 | 0.766 | 0.314559 | 0.373770 | -0.059211 | 157 | 43 | 0.960000 | 1.000000 |
| utility damage | 10 | 0.038 | 0.480843 | 0.471706 | 0.009137 | 1 | 9 | 1.000000 | 1.000000 |
| active smoke/inferno | 200 | 0.766 | 0.314559 | 0.373770 | -0.059211 | 157 | 43 | 0.960000 | 1.000000 |
| recent utility last 5s | 10 | 0.038 | 0.417154 | 0.477597 | -0.060442 | 10 | 0 | 1.000000 | 1.000000 |
| flash effect present | 261 | 1.000 | 0.295235 | 0.353387 | -0.058152 | 207 | 54 | 0.946360 | 1.000000 |

## Active Smoke/Inferno Intervals

- `8.5s` - `42.0s`, rows `68`
- `48.5s` - `101.5s`, rows `107`
- `103.0s` - `108.0s`, rows `11`
- `118.0s` - `124.5s`, rows `14`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `121.5`, LSTM `0.1706`, XGBoost `0.4770`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `121.0`, LSTM `0.1584`, XGBoost `0.4627`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `120.5`, LSTM `0.1405`, XGBoost `0.4409`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `122.0`, LSTM `0.1904`, XGBoost `0.4858`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `122.5`, LSTM `0.2022`, XGBoost `0.4674`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `123.0`, LSTM `0.2082`, XGBoost `0.4660`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `123.5`, LSTM `0.2285`, XGBoost `0.4786`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `124.0`, LSTM `0.2559`, XGBoost `0.4786`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `124.5`, LSTM `0.2573`, XGBoost `0.4786`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `83.0`, LSTM `0.2716`, XGBoost `0.4522`, closer `lstm`, smoke `4`, inferno `1`, utility_damage `0.0`, recent_utility `0`
