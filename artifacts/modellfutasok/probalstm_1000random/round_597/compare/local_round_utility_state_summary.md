# Local Round Utility State Analysis

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-falcons-vs-legacy-bo3-ryWGopRV1OfbL288nR6Rql/falcons-vs-legacy-m1-inferno.csv`
- round_num: `4`
- rows: `143`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 143 | 1.000 | 0.510189 | 0.401991 | 0.108199 | 135 | 8 | 0.643357 | 0.027972 |
| active/recent utility | 143 | 1.000 | 0.510189 | 0.401991 | 0.108199 | 135 | 8 | 0.643357 | 0.027972 |
| strong utility action | 115 | 0.804 | 0.492581 | 0.367520 | 0.125061 | 111 | 4 | 0.556522 | 0.000000 |
| utility damage | 10 | 0.070 | 0.626837 | 0.494600 | 0.132238 | 10 | 0 | 1.000000 | 0.000000 |
| active smoke/inferno | 115 | 0.804 | 0.492581 | 0.367520 | 0.125061 | 111 | 4 | 0.556522 | 0.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 143 | 1.000 | 0.510189 | 0.401991 | 0.108199 | 135 | 8 | 0.643357 | 0.027972 |

## Active Smoke/Inferno Intervals

- `9.5s` - `66.5s`, rows `115`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `57.5`, LSTM `0.4682`, XGBoost `0.1777`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `54.5`, LSTM `0.4993`, XGBoost `0.2158`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `53.0`, LSTM `0.4958`, XGBoost `0.2131`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `52.5`, LSTM `0.4956`, XGBoost `0.2131`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `55.0`, LSTM `0.4964`, XGBoost `0.2158`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `57.0`, LSTM `0.4586`, XGBoost `0.1796`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `54.0`, LSTM `0.4940`, XGBoost `0.2172`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `53.5`, LSTM `0.4938`, XGBoost `0.2172`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `52.0`, LSTM `0.4707`, XGBoost `0.2131`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `50.5`, LSTM `0.4686`, XGBoost `0.2131`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
