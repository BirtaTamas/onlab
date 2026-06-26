# Local Round Utility State Analysis

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-falcons-vs-pain-bo3-6mWraId8pA69o5etX6dmBT/falcons-vs-pain-m1-inferno.csv`
- round_num: `11`
- rows: `184`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 184 | 1.000 | 0.756661 | 0.743867 | 0.012794 | 98 | 86 | 1.000000 | 0.945652 |
| active/recent utility | 184 | 1.000 | 0.756661 | 0.743867 | 0.012794 | 98 | 86 | 1.000000 | 0.945652 |
| strong utility action | 163 | 0.886 | 0.786576 | 0.774920 | 0.011656 | 77 | 86 | 1.000000 | 0.938650 |
| utility damage | 20 | 0.109 | 0.703645 | 0.691754 | 0.011890 | 10 | 10 | 1.000000 | 1.000000 |
| active smoke/inferno | 163 | 0.886 | 0.786576 | 0.774920 | 0.011656 | 77 | 86 | 1.000000 | 0.938650 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 184 | 1.000 | 0.756661 | 0.743867 | 0.012794 | 98 | 86 | 1.000000 | 0.945652 |

## Active Smoke/Inferno Intervals

- `10.5s` - `91.5s`, rows `163`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `67.5`, LSTM `0.8697`, XGBoost `0.7459`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `67.0`, LSTM `0.8641`, XGBoost `0.7447`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `62.0`, LSTM `0.8679`, XGBoost `0.7501`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `66.5`, LSTM `0.8619`, XGBoost `0.7447`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `68.0`, LSTM `0.8585`, XGBoost `0.7446`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `28.5`, LSTM `0.6140`, XGBoost `0.5022`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `30.0`, recent_utility `0`
- seconds `73.5`, LSTM `0.8581`, XGBoost `0.7489`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `66.0`, LSTM `0.8539`, XGBoost `0.7447`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `68.5`, LSTM `0.8525`, XGBoost `0.7434`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `72.0`, LSTM `0.8550`, XGBoost `0.7468`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
