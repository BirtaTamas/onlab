# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-liquid-vs-3dmax-bo3-k7r_vGkiL4eRhxKdRPUZx1/liquid-vs-3dmax-m3-anubis.csv`
- round_num: `2`
- rows: `151`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 151 | 1.000 | 0.013384 | 0.054478 | -0.041093 | 151 | 0 | 1.000000 | 1.000000 |
| active/recent utility | 123 | 0.815 | 0.013523 | 0.055818 | -0.042295 | 123 | 0 | 1.000000 | 1.000000 |
| strong utility action | 123 | 0.815 | 0.013523 | 0.055818 | -0.042295 | 123 | 0 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 113 | 0.748 | 0.013307 | 0.055482 | -0.042174 | 113 | 0 | 1.000000 | 1.000000 |
| recent utility last 5s | 10 | 0.066 | 0.015955 | 0.059614 | -0.043659 | 10 | 0 | 1.000000 | 1.000000 |
| flash effect present | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |

## Active Smoke/Inferno Intervals

- `14.5s` - `48.5s`, rows `69`
- `52.5s` - `74.0s`, rows `44`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `22.5`, LSTM `0.0074`, XGBoost `0.0659`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `24.0`, LSTM `0.0082`, XGBoost `0.0665`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `24.5`, LSTM `0.0082`, XGBoost `0.0665`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `23.5`, LSTM `0.0081`, XGBoost `0.0660`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `37.5`, LSTM `0.0089`, XGBoost `0.0667`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `40.0`, LSTM `0.0090`, XGBoost `0.0667`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `23.0`, LSTM `0.0083`, XGBoost `0.0659`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `28.0`, LSTM `0.0092`, XGBoost `0.0667`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `25.0`, LSTM `0.0091`, XGBoost `0.0665`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `37.0`, LSTM `0.0093`, XGBoost `0.0667`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
