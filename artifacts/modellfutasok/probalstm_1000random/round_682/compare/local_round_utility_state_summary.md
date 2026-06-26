# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-astralis-vs-fluxo-bo3-sWQe-jgKNP3vaioXQrjxgB/astralis-vs-fluxo-m3-nuke.csv`
- round_num: `4`
- rows: `233`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 233 | 1.000 | 0.181318 | 0.178901 | 0.002417 | 191 | 42 | 0.931330 | 0.819742 |
| active/recent utility | 233 | 1.000 | 0.181318 | 0.178901 | 0.002417 | 191 | 42 | 0.931330 | 0.819742 |
| strong utility action | 138 | 0.592 | 0.242604 | 0.228094 | 0.014510 | 96 | 42 | 0.884058 | 0.826087 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 138 | 0.592 | 0.242604 | 0.228094 | 0.014510 | 96 | 42 | 0.884058 | 0.826087 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 233 | 1.000 | 0.181318 | 0.178901 | 0.002417 | 191 | 42 | 0.931330 | 0.819742 |

## Active Smoke/Inferno Intervals

- `9.0s` - `70.5s`, rows `124`
- `79.0s` - `85.5s`, rows `14`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `28.5`, LSTM `0.5099`, XGBoost `0.3364`, closer `xgboost`, smoke `6`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `29.0`, LSTM `0.5068`, XGBoost `0.3364`, closer `xgboost`, smoke `6`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `28.0`, LSTM `0.5023`, XGBoost `0.3365`, closer `xgboost`, smoke `6`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `30.0`, LSTM `0.5025`, XGBoost `0.3369`, closer `xgboost`, smoke `6`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `30.5`, LSTM `0.4996`, XGBoost `0.3366`, closer `xgboost`, smoke `6`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `29.5`, LSTM `0.4957`, XGBoost `0.3369`, closer `xgboost`, smoke `6`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `31.5`, LSTM `0.4885`, XGBoost `0.3366`, closer `xgboost`, smoke `6`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `27.5`, LSTM `0.4865`, XGBoost `0.3365`, closer `xgboost`, smoke `6`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `31.0`, LSTM `0.4830`, XGBoost `0.3366`, closer `xgboost`, smoke `6`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `27.0`, LSTM `0.4790`, XGBoost `0.3365`, closer `xgboost`, smoke `6`, inferno `1`, utility_damage `0.0`, recent_utility `0`
