# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-legacy-vs-gentle-mates-bo3-EYv8hp-oY0glsojznK6Qby/legacy-vs-gentle-mates-m2-mirage.csv`
- round_num: `11`
- rows: `207`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 207 | 1.000 | 0.647102 | 0.717168 | -0.070066 | 4 | 203 | 0.937198 | 1.000000 |
| active/recent utility | 207 | 1.000 | 0.647102 | 0.717168 | -0.070066 | 4 | 203 | 0.937198 | 1.000000 |
| strong utility action | 189 | 0.913 | 0.648428 | 0.723271 | -0.074843 | 4 | 185 | 0.962963 | 1.000000 |
| utility damage | 11 | 0.053 | 0.781821 | 0.852383 | -0.070562 | 0 | 11 | 1.000000 | 1.000000 |
| active smoke/inferno | 189 | 0.913 | 0.648428 | 0.723271 | -0.074843 | 4 | 185 | 0.962963 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 207 | 1.000 | 0.647102 | 0.717168 | -0.070066 | 4 | 203 | 0.937198 | 1.000000 |

## Active Smoke/Inferno Intervals

- `6.0s` - `100.0s`, rows `189`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `49.5`, LSTM `0.5572`, XGBoost `0.7375`, closer `xgboost`, smoke `1`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `50.0`, LSTM `0.5574`, XGBoost `0.7361`, closer `xgboost`, smoke `1`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `49.0`, LSTM `0.5606`, XGBoost `0.7370`, closer `xgboost`, smoke `1`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `48.5`, LSTM `0.5684`, XGBoost `0.7370`, closer `xgboost`, smoke `1`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `47.0`, LSTM `0.5713`, XGBoost `0.7363`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `48.0`, LSTM `0.5727`, XGBoost `0.7370`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `54.0`, LSTM `0.5711`, XGBoost `0.7354`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `51.0`, LSTM `0.5719`, XGBoost `0.7354`, closer `xgboost`, smoke `1`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `50.5`, LSTM `0.5724`, XGBoost `0.7356`, closer `xgboost`, smoke `1`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `55.0`, LSTM `0.5731`, XGBoost `0.7354`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
