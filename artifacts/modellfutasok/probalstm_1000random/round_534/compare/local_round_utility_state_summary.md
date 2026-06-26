# Local Round Utility State Analysis

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-falcons-vs-pain-bo3-6mWraId8pA69o5etX6dmBT/falcons-vs-pain-m1-inferno.csv`
- round_num: `14`
- rows: `227`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 227 | 1.000 | 0.159360 | 0.181694 | -0.022334 | 170 | 57 | 1.000000 | 1.000000 |
| active/recent utility | 227 | 1.000 | 0.159360 | 0.181694 | -0.022334 | 170 | 57 | 1.000000 | 1.000000 |
| strong utility action | 169 | 0.744 | 0.159879 | 0.174829 | -0.014950 | 118 | 51 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 153 | 0.674 | 0.146131 | 0.155874 | -0.009743 | 102 | 51 | 1.000000 | 1.000000 |
| recent utility last 5s | 16 | 0.070 | 0.291349 | 0.356092 | -0.064743 | 16 | 0 | 1.000000 | 1.000000 |
| flash effect present | 227 | 1.000 | 0.159360 | 0.181694 | -0.022334 | 170 | 57 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `11.0s` - `18.0s`, rows `15`
- `31.5s` - `100.0s`, rows `138`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `3.0`, LSTM `0.2535`, XGBoost `0.3643`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `3`
- seconds `17.5`, LSTM `0.2698`, XGBoost `0.3707`, closer `lstm`, smoke `0`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `0.5`, LSTM `0.2332`, XGBoost `0.3311`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `2`
- seconds `3.5`, LSTM `0.2731`, XGBoost `0.3703`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `4`
- seconds `11.0`, LSTM `0.2752`, XGBoost `0.3682`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `4.0`, LSTM `0.2779`, XGBoost `0.3703`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `4`
- seconds `1.5`, LSTM `0.2374`, XGBoost `0.3299`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `3`
- seconds `11.5`, LSTM `0.2758`, XGBoost `0.3682`, closer `lstm`, smoke `0`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `2.0`, LSTM `0.2397`, XGBoost `0.3299`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `3`
- seconds `2.5`, LSTM `0.2397`, XGBoost `0.3264`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `3`
