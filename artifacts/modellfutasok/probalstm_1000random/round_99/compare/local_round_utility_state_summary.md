# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-vitality-vs-3dmax-bo3-SFueR4Yd1u5-bIhh5XKwOq/vitality-vs-3dmax-m2-dust2.csv`
- round_num: `11`
- rows: `197`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 197 | 1.000 | 0.131705 | 0.174043 | -0.042337 | 196 | 1 | 0.979695 | 0.888325 |
| active/recent utility | 197 | 1.000 | 0.131705 | 0.174043 | -0.042337 | 196 | 1 | 0.979695 | 0.888325 |
| strong utility action | 154 | 0.782 | 0.153673 | 0.202703 | -0.049030 | 153 | 1 | 0.974026 | 0.857143 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 154 | 0.782 | 0.153673 | 0.202703 | -0.049030 | 153 | 1 | 0.974026 | 0.857143 |
| recent utility last 5s | 10 | 0.051 | 0.462427 | 0.569093 | -0.106666 | 10 | 0 | 0.700000 | 0.100000 |
| flash effect present | 197 | 1.000 | 0.131705 | 0.174043 | -0.042337 | 196 | 1 | 0.979695 | 0.888325 |

## Active Smoke/Inferno Intervals

- `2.5s` - `73.5s`, rows `143`
- `84.5s` - `89.5s`, rows `11`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `14.0`, LSTM `0.5418`, XGBoost `0.7241`, closer `lstm`, smoke `3`, inferno `2`, utility_damage `0.0`, recent_utility `1`
- seconds `8.0`, LSTM `0.3021`, XGBoost `0.4829`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `9.5`, LSTM `0.3040`, XGBoost `0.4838`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `8.5`, LSTM `0.3075`, XGBoost `0.4846`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `14.5`, LSTM `0.5524`, XGBoost `0.7241`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `1`
- seconds `15.0`, LSTM `0.5540`, XGBoost `0.7215`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `1`
- seconds `9.0`, LSTM `0.3217`, XGBoost `0.4838`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `10.0`, LSTM `0.3271`, XGBoost `0.4850`, closer `lstm`, smoke `2`, inferno `3`, utility_damage `0.0`, recent_utility `0`
- seconds `10.5`, LSTM `0.3287`, XGBoost `0.4850`, closer `lstm`, smoke `3`, inferno `3`, utility_damage `0.0`, recent_utility `0`
- seconds `29.0`, LSTM `0.2735`, XGBoost `0.4235`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `56.0`, recent_utility `0`
