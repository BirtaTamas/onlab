# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-astralis-vs-ence-bo3-avzZyhQ46OR9GyYE_6NeM7/astralis-vs-ence-m3-overpass.csv`
- round_num: `19`
- rows: `211`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 211 | 1.000 | 0.018967 | 0.058588 | -0.039621 | 202 | 9 | 1.000000 | 1.000000 |
| active/recent utility | 211 | 1.000 | 0.018967 | 0.058588 | -0.039621 | 202 | 9 | 1.000000 | 1.000000 |
| strong utility action | 91 | 0.431 | 0.020460 | 0.059134 | -0.038674 | 84 | 7 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 91 | 0.431 | 0.020460 | 0.059134 | -0.038674 | 84 | 7 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 211 | 1.000 | 0.018967 | 0.058588 | -0.039621 | 202 | 9 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `9.5s` - `40.5s`, rows `63`
- `51.5s` - `58.0s`, rows `14`
- `97.5s` - `104.0s`, rows `14`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `11.5`, LSTM `0.0239`, XGBoost `0.0817`, closer `lstm`, smoke `1`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `11.0`, LSTM `0.0231`, XGBoost `0.0807`, closer `lstm`, smoke `1`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `10.0`, LSTM `0.0226`, XGBoost `0.0801`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `10.5`, LSTM `0.0235`, XGBoost `0.0802`, closer `lstm`, smoke `1`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `9.5`, LSTM `0.0233`, XGBoost `0.0791`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `35.5`, LSTM `0.0106`, XGBoost `0.0644`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `40.5`, LSTM `0.0136`, XGBoost `0.0667`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `35.0`, LSTM `0.0114`, XGBoost `0.0644`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `36.0`, LSTM `0.0122`, XGBoost `0.0644`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `36.5`, LSTM `0.0131`, XGBoost `0.0644`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
