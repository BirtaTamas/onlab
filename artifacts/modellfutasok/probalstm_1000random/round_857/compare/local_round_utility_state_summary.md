# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-lynn-vision-vs-housebets-bo3-GrWDn9AJOxYQcZMXkSI-Tw/lynn-vision-vs-housebets-m1-inferno.csv`
- round_num: `11`
- rows: `122`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 122 | 1.000 | 0.469568 | 0.411159 | 0.058409 | 81 | 41 | 0.450820 | 0.450820 |
| active/recent utility | 122 | 1.000 | 0.469568 | 0.411159 | 0.058409 | 81 | 41 | 0.450820 | 0.450820 |
| strong utility action | 104 | 0.852 | 0.425912 | 0.377597 | 0.048315 | 63 | 41 | 0.355769 | 0.355769 |
| utility damage | 27 | 0.221 | 0.446296 | 0.356743 | 0.089553 | 25 | 2 | 0.222222 | 0.222222 |
| active smoke/inferno | 104 | 0.852 | 0.425912 | 0.377597 | 0.048315 | 63 | 41 | 0.355769 | 0.355769 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 122 | 1.000 | 0.469568 | 0.411159 | 0.058409 | 81 | 41 | 0.450820 | 0.450820 |

## Active Smoke/Inferno Intervals

- `9.0s` - `60.5s`, rows `104`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `52.5`, LSTM `0.4909`, XGBoost `0.3224`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `50.0`, LSTM `0.4779`, XGBoost `0.3112`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `1.0`, recent_utility `0`
- seconds `50.5`, LSTM `0.4867`, XGBoost `0.3224`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `1.0`, recent_utility `0`
- seconds `53.0`, LSTM `0.4865`, XGBoost `0.3224`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `52.0`, LSTM `0.4849`, XGBoost `0.3224`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `53.5`, LSTM `0.4813`, XGBoost `0.3224`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `13.0`, LSTM `0.7192`, XGBoost `0.5656`, closer `lstm`, smoke `1`, inferno `3`, utility_damage `16.0`, recent_utility `0`
- seconds `49.5`, LSTM `0.4609`, XGBoost `0.3112`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `1.0`, recent_utility `0`
- seconds `14.0`, LSTM `0.7027`, XGBoost `0.5544`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `48.0`, recent_utility `0`
- seconds `51.5`, LSTM `0.4701`, XGBoost `0.3224`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
