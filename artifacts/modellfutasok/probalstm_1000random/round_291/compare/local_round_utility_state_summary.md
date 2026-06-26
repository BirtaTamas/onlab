# Local Round Utility State Analysis

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-falcons-vs-pain-bo3-6mWraId8pA69o5etX6dmBT/falcons-vs-pain-m1-inferno.csv`
- round_num: `17`
- rows: `232`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 232 | 1.000 | 0.367360 | 0.464308 | -0.096949 | 207 | 25 | 0.862069 | 0.719828 |
| active/recent utility | 232 | 1.000 | 0.367360 | 0.464308 | -0.096949 | 207 | 25 | 0.862069 | 0.719828 |
| strong utility action | 219 | 0.944 | 0.369407 | 0.471293 | -0.101886 | 200 | 19 | 0.881279 | 0.730594 |
| utility damage | 10 | 0.043 | 0.526524 | 0.532304 | -0.005780 | 4 | 6 | 0.200000 | 0.100000 |
| active smoke/inferno | 209 | 0.901 | 0.374732 | 0.471038 | -0.096306 | 190 | 19 | 0.875598 | 0.717703 |
| recent utility last 5s | 10 | 0.043 | 0.258110 | 0.476620 | -0.218510 | 10 | 0 | 1.000000 | 1.000000 |
| flash effect present | 232 | 1.000 | 0.367360 | 0.464308 | -0.096949 | 207 | 25 | 0.862069 | 0.719828 |

## Active Smoke/Inferno Intervals

- `6.5s` - `87.0s`, rows `162`
- `90.5s` - `113.5s`, rows `47`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `56.5`, LSTM `0.2359`, XGBoost `0.4887`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `2.5`, LSTM `0.2452`, XGBoost `0.4800`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `3.0`, LSTM `0.2403`, XGBoost `0.4749`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `4.0`, LSTM `0.2414`, XGBoost `0.4749`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `3.5`, LSTM `0.2432`, XGBoost `0.4749`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `56.0`, LSTM `0.2644`, XGBoost `0.4887`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `2.0`, LSTM `0.2573`, XGBoost `0.4800`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `1.5`, LSTM `0.2584`, XGBoost `0.4800`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `1.0`, LSTM `0.2626`, XGBoost `0.4814`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `55.5`, LSTM `0.2715`, XGBoost `0.4887`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
