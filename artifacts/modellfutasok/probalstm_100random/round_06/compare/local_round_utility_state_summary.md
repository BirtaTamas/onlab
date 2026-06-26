# Local Round Utility State Analysis

- csv_path: `processed_full/blast_bounty_season_1/blast-bounty-2025-season-1-flyquest-vs-mibr-bo3-qPrK-wzQgATa8KQ5HjYeOS/flyquest-vs-mibr-m1-nuke.csv`
- round_num: `3`
- rows: `272`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 272 | 1.000 | 0.247659 | 0.299609 | -0.051949 | 267 | 5 | 0.816176 | 0.514706 |
| active/recent utility | 272 | 1.000 | 0.247659 | 0.299609 | -0.051949 | 267 | 5 | 0.816176 | 0.514706 |
| strong utility action | 161 | 0.592 | 0.371513 | 0.444001 | -0.072489 | 156 | 5 | 0.689441 | 0.279503 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 161 | 0.592 | 0.371513 | 0.444001 | -0.072489 | 156 | 5 | 0.689441 | 0.279503 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 272 | 1.000 | 0.247659 | 0.299609 | -0.051949 | 267 | 5 | 0.816176 | 0.514706 |

## Active Smoke/Inferno Intervals

- `8.0s` - `88.0s`, rows `161`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `67.0`, LSTM `0.1040`, XGBoost `0.3702`, closer `lstm`, smoke `3`, inferno `3`, utility_damage `10.0`, recent_utility `0`
- seconds `68.5`, LSTM `0.1140`, XGBoost `0.3688`, closer `lstm`, smoke `3`, inferno `3`, utility_damage `7.0`, recent_utility `0`
- seconds `67.5`, LSTM `0.1188`, XGBoost `0.3696`, closer `lstm`, smoke `3`, inferno `3`, utility_damage `10.0`, recent_utility `0`
- seconds `66.5`, LSTM `0.1470`, XGBoost `0.3702`, closer `lstm`, smoke `4`, inferno `3`, utility_damage `10.0`, recent_utility `0`
- seconds `68.0`, LSTM `0.1492`, XGBoost `0.3688`, closer `lstm`, smoke `3`, inferno `3`, utility_damage `10.0`, recent_utility `0`
- seconds `69.0`, LSTM `0.1611`, XGBoost `0.3734`, closer `lstm`, smoke `3`, inferno `3`, utility_damage `0.0`, recent_utility `0`
- seconds `66.0`, LSTM `0.1496`, XGBoost `0.3596`, closer `lstm`, smoke `3`, inferno `3`, utility_damage `10.0`, recent_utility `0`
- seconds `48.5`, LSTM `0.3525`, XGBoost `0.5462`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `49.0`, LSTM `0.3906`, XGBoost `0.5462`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `48.0`, LSTM `0.3910`, XGBoost `0.5418`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
