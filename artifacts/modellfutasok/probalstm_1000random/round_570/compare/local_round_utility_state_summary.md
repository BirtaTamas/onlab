# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-falcons-vs-3dmax-bo3-XHM3Ovc8L9TfLFTYQFrGdT/falcons-vs-3dmax-m3-dust2.csv`
- round_num: `1`
- rows: `126`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 126 | 1.000 | 0.220216 | 0.251217 | -0.031001 | 92 | 34 | 0.730159 | 0.769841 |
| active/recent utility | 126 | 1.000 | 0.220216 | 0.251217 | -0.031001 | 92 | 34 | 0.730159 | 0.769841 |
| strong utility action | 93 | 0.738 | 0.213487 | 0.251502 | -0.038015 | 69 | 24 | 0.795699 | 0.849462 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 93 | 0.738 | 0.213487 | 0.251502 | -0.038015 | 69 | 24 | 0.795699 | 0.849462 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 126 | 1.000 | 0.220216 | 0.251217 | -0.031001 | 92 | 34 | 0.730159 | 0.769841 |

## Active Smoke/Inferno Intervals

- `7.5s` - `53.5s`, rows `93`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `31.0`, LSTM `0.0338`, XGBoost `0.1877`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `35.0`, LSTM `0.0343`, XGBoost `0.1784`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `31.5`, LSTM `0.0473`, XGBoost `0.1911`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `34.5`, LSTM `0.0351`, XGBoost `0.1779`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `34.0`, LSTM `0.0366`, XGBoost `0.1779`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `30.5`, LSTM `0.0358`, XGBoost `0.1761`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `33.0`, LSTM `0.0319`, XGBoost `0.1710`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `33.5`, LSTM `0.0357`, XGBoost `0.1746`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `30.0`, LSTM `0.0429`, XGBoost `0.1633`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `29.5`, LSTM `0.0383`, XGBoost `0.1568`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
