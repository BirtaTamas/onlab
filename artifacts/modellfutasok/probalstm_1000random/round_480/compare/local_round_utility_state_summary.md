# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-vitality-vs-the-mongolz-bo3-7VmOOQFfF_Xgx4vOG4cYIY/vitality-vs-the-mongolz-m3-inferno.csv`
- round_num: `3`
- rows: `278`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 278 | 1.000 | 0.015260 | 0.030447 | -0.015187 | 267 | 11 | 1.000000 | 1.000000 |
| active/recent utility | 278 | 1.000 | 0.015260 | 0.030447 | -0.015187 | 267 | 11 | 1.000000 | 1.000000 |
| strong utility action | 140 | 0.504 | 0.017140 | 0.030257 | -0.013117 | 129 | 11 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 140 | 0.504 | 0.017140 | 0.030257 | -0.013117 | 129 | 11 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 278 | 1.000 | 0.015260 | 0.030447 | -0.015187 | 267 | 11 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `10.0s` - `17.5s`, rows `16`
- `22.5s` - `31.0s`, rows `18`
- `62.0s` - `114.5s`, rows `106`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `10.0`, LSTM `0.0228`, XGBoost `0.0640`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `10.5`, LSTM `0.0252`, XGBoost `0.0644`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `11.5`, LSTM `0.0270`, XGBoost `0.0657`, closer `lstm`, smoke `0`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `14.0`, LSTM `0.0387`, XGBoost `0.0770`, closer `lstm`, smoke `0`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `11.0`, LSTM `0.0274`, XGBoost `0.0656`, closer `lstm`, smoke `0`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `12.5`, LSTM `0.0364`, XGBoost `0.0742`, closer `lstm`, smoke `0`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `15.5`, LSTM `0.0400`, XGBoost `0.0776`, closer `lstm`, smoke `0`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `15.0`, LSTM `0.0423`, XGBoost `0.0776`, closer `lstm`, smoke `0`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `12.0`, LSTM `0.0312`, XGBoost `0.0657`, closer `lstm`, smoke `0`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `14.5`, LSTM `0.0438`, XGBoost `0.0776`, closer `lstm`, smoke `0`, inferno `2`, utility_damage `0.0`, recent_utility `0`
