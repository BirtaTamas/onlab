# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-g2-vs-fluxo-bo3-IhqycqXYyOA3DyfY0xuGyX/g2-vs-fluxo-m2-inferno.csv`
- round_num: `8`
- rows: `169`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 169 | 1.000 | 0.092410 | 0.126893 | -0.034483 | 166 | 3 | 1.000000 | 1.000000 |
| active/recent utility | 169 | 1.000 | 0.092410 | 0.126893 | -0.034483 | 166 | 3 | 1.000000 | 1.000000 |
| strong utility action | 125 | 0.740 | 0.073202 | 0.105176 | -0.031974 | 122 | 3 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 125 | 0.740 | 0.073202 | 0.105176 | -0.031974 | 122 | 3 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 169 | 1.000 | 0.092410 | 0.126893 | -0.034483 | 166 | 3 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `9.5s` - `71.5s`, rows `125`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `12.0`, LSTM `0.2623`, XGBoost `0.4400`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `12.5`, LSTM `0.2831`, XGBoost `0.4400`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `11.5`, LSTM `0.2841`, XGBoost `0.4400`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `17.5`, LSTM `0.2965`, XGBoost `0.4401`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `18.0`, LSTM `0.2969`, XGBoost `0.4397`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `13.5`, LSTM `0.2995`, XGBoost `0.4404`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `16.5`, LSTM `0.2933`, XGBoost `0.4291`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `17.0`, LSTM `0.3040`, XGBoost `0.4380`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `15.5`, LSTM `0.2969`, XGBoost `0.4284`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `16.0`, LSTM `0.2983`, XGBoost `0.4290`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
