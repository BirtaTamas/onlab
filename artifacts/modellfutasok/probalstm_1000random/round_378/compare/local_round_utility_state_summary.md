# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-liquid-vs-3dmax-bo3-k7r_vGkiL4eRhxKdRPUZx1/liquid-vs-3dmax-m2-ancient.csv`
- round_num: `14`
- rows: `154`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 154 | 1.000 | 0.011434 | 0.032516 | -0.021082 | 150 | 4 | 1.000000 | 1.000000 |
| active/recent utility | 154 | 1.000 | 0.011434 | 0.032516 | -0.021082 | 150 | 4 | 1.000000 | 1.000000 |
| strong utility action | 120 | 0.779 | 0.011673 | 0.034177 | -0.022504 | 119 | 1 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 110 | 0.714 | 0.011427 | 0.033962 | -0.022535 | 109 | 1 | 1.000000 | 1.000000 |
| recent utility last 5s | 10 | 0.065 | 0.014375 | 0.036545 | -0.022169 | 10 | 0 | 1.000000 | 1.000000 |
| flash effect present | 154 | 1.000 | 0.011434 | 0.032516 | -0.021082 | 150 | 4 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `7.0s` - `39.5s`, rows `66`
- `50.5s` - `72.0s`, rows `44`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `19.0`, LSTM `0.0072`, XGBoost `0.0357`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `18.5`, LSTM `0.0075`, XGBoost `0.0357`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `16.0`, LSTM `0.0078`, XGBoost `0.0358`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `21.5`, LSTM `0.0077`, XGBoost `0.0357`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `1.0`, LSTM `0.0106`, XGBoost `0.0385`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `16.5`, LSTM `0.0079`, XGBoost `0.0357`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `15.5`, LSTM `0.0081`, XGBoost `0.0358`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `19.5`, LSTM `0.0082`, XGBoost `0.0357`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `17.5`, LSTM `0.0086`, XGBoost `0.0357`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `17.0`, LSTM `0.0086`, XGBoost `0.0357`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
