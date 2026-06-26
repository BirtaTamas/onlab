# Local Round Utility State Analysis

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-natus-vincere-vs-3dmax-bo3-JB3JZO-5zNCohi5tAgyHtq/natus-vincere-vs-3dmax-m2-inferno.csv`
- round_num: `7`
- rows: `167`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 167 | 1.000 | 0.476756 | 0.557968 | -0.081212 | 160 | 7 | 0.622754 | 0.526946 |
| active/recent utility | 167 | 1.000 | 0.476756 | 0.557968 | -0.081212 | 160 | 7 | 0.622754 | 0.526946 |
| strong utility action | 120 | 0.719 | 0.484593 | 0.579441 | -0.094848 | 120 | 0 | 0.583333 | 0.483333 |
| utility damage | 20 | 0.120 | 0.608740 | 0.714693 | -0.105954 | 20 | 0 | 0.250000 | 0.000000 |
| active smoke/inferno | 110 | 0.659 | 0.496731 | 0.589197 | -0.092466 | 110 | 0 | 0.545455 | 0.436364 |
| recent utility last 5s | 10 | 0.060 | 0.351082 | 0.472133 | -0.121050 | 10 | 0 | 1.000000 | 1.000000 |
| flash effect present | 167 | 1.000 | 0.476756 | 0.557968 | -0.081212 | 160 | 7 | 0.622754 | 0.526946 |

## Active Smoke/Inferno Intervals

- `10.5s` - `37.5s`, rows `55`
- `50.5s` - `72.0s`, rows `44`
- `75.5s` - `80.5s`, rows `11`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `26.0`, LSTM `0.5554`, XGBoost `0.7755`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `26.5`, LSTM `0.5619`, XGBoost `0.7755`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `25.5`, LSTM `0.5681`, XGBoost `0.7755`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `27.0`, LSTM `0.5727`, XGBoost `0.7755`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `28.0`, LSTM `0.5742`, XGBoost `0.7739`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `27.5`, LSTM `0.5766`, XGBoost `0.7755`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `19.0`, LSTM `0.5440`, XGBoost `0.7394`, closer `lstm`, smoke `4`, inferno `1`, utility_damage `8.0`, recent_utility `0`
- seconds `25.0`, LSTM `0.5815`, XGBoost `0.7755`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `18.5`, LSTM `0.5458`, XGBoost `0.7371`, closer `lstm`, smoke `4`, inferno `1`, utility_damage `8.0`, recent_utility `0`
- seconds `24.5`, LSTM `0.5847`, XGBoost `0.7755`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
