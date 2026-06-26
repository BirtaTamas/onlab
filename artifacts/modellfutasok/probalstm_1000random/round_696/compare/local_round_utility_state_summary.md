# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-legacy-vs-gentle-mates-bo3-EYv8hp-oY0glsojznK6Qby/legacy-vs-gentle-mates-m2-mirage.csv`
- round_num: `15`
- rows: `203`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 203 | 1.000 | 0.273364 | 0.347528 | -0.074164 | 195 | 8 | 0.965517 | 0.684729 |
| active/recent utility | 203 | 1.000 | 0.273364 | 0.347528 | -0.074164 | 195 | 8 | 0.965517 | 0.684729 |
| strong utility action | 141 | 0.695 | 0.271293 | 0.344902 | -0.073609 | 140 | 1 | 0.950355 | 0.751773 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 141 | 0.695 | 0.271293 | 0.344902 | -0.073609 | 140 | 1 | 0.950355 | 0.751773 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 203 | 1.000 | 0.273364 | 0.347528 | -0.074164 | 195 | 8 | 0.965517 | 0.684729 |

## Active Smoke/Inferno Intervals

- `11.0s` - `56.0s`, rows `91`
- `71.0s` - `95.5s`, rows `50`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `92.5`, LSTM `0.1373`, XGBoost `0.3238`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `54.0`, LSTM `0.3314`, XGBoost `0.5175`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `53.5`, LSTM `0.3329`, XGBoost `0.5175`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `53.0`, LSTM `0.3462`, XGBoost `0.5175`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `54.5`, LSTM `0.3480`, XGBoost `0.5175`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `90.0`, LSTM `0.1472`, XGBoost `0.3149`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `52.5`, LSTM `0.3503`, XGBoost `0.5175`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `55.0`, LSTM `0.3548`, XGBoost `0.5175`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `89.5`, LSTM `0.1503`, XGBoost `0.3129`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `89.0`, LSTM `0.1547`, XGBoost `0.3135`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
