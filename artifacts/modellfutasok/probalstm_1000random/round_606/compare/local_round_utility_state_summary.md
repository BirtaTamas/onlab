# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-lynn-vision-vs-housebets-bo3-GrWDn9AJOxYQcZMXkSI-Tw/lynn-vision-vs-housebets-m1-inferno.csv`
- round_num: `9`
- rows: `235`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 235 | 1.000 | 0.600654 | 0.539286 | 0.061368 | 200 | 35 | 0.846809 | 0.693617 |
| active/recent utility | 235 | 1.000 | 0.600654 | 0.539286 | 0.061368 | 200 | 35 | 0.846809 | 0.693617 |
| strong utility action | 170 | 0.723 | 0.592056 | 0.534922 | 0.057134 | 144 | 26 | 0.900000 | 0.688235 |
| utility damage | 16 | 0.068 | 0.683707 | 0.568683 | 0.115024 | 16 | 0 | 1.000000 | 1.000000 |
| active smoke/inferno | 170 | 0.723 | 0.592056 | 0.534922 | 0.057134 | 144 | 26 | 0.900000 | 0.688235 |
| recent utility last 5s | 10 | 0.043 | 0.528742 | 0.370090 | 0.158652 | 10 | 0 | 0.700000 | 0.000000 |
| flash effect present | 235 | 1.000 | 0.600654 | 0.539286 | 0.061368 | 200 | 35 | 0.846809 | 0.693617 |

## Active Smoke/Inferno Intervals

- `10.0s` - `34.0s`, rows `49`
- `39.5s` - `99.5s`, rows `121`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `99.0`, LSTM `0.5272`, XGBoost `0.2208`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `99.5`, LSTM `0.4994`, XGBoost `0.2208`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `62.5`, LSTM `0.5811`, XGBoost `0.3733`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `63.5`, LSTM `0.5577`, XGBoost `0.3536`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `63.0`, LSTM `0.5521`, XGBoost `0.3499`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `62.0`, LSTM `0.6012`, XGBoost `0.4063`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `64.0`, LSTM `0.5467`, XGBoost `0.3579`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `61.5`, LSTM `0.6351`, XGBoost `0.4503`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `64.5`, LSTM `0.5211`, XGBoost `0.3601`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `54.0`, LSTM `0.1829`, XGBoost `0.3432`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
