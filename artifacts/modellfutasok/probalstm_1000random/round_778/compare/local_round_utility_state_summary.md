# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-heroic-vs-3dmax-bo3-Dgk7HiwYvj5CMwMpEHLxHJ/heroic-vs-3dmax-m1-nuke.csv`
- round_num: `18`
- rows: `272`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 272 | 1.000 | 0.524636 | 0.479992 | 0.044644 | 167 | 105 | 0.264706 | 0.283088 |
| active/recent utility | 272 | 1.000 | 0.524636 | 0.479992 | 0.044644 | 167 | 105 | 0.264706 | 0.283088 |
| strong utility action | 168 | 0.618 | 0.553875 | 0.571976 | -0.018101 | 143 | 25 | 0.071429 | 0.089286 |
| utility damage | 31 | 0.114 | 0.554317 | 0.603379 | -0.049062 | 31 | 0 | 0.000000 | 0.000000 |
| active smoke/inferno | 168 | 0.618 | 0.553875 | 0.571976 | -0.018101 | 143 | 25 | 0.071429 | 0.089286 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 272 | 1.000 | 0.524636 | 0.479992 | 0.044644 | 167 | 105 | 0.264706 | 0.283088 |

## Active Smoke/Inferno Intervals

- `7.5s` - `57.5s`, rows `101`
- `60.5s` - `86.5s`, rows `53`
- `98.0s` - `104.5s`, rows `14`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `100.5`, LSTM `0.5042`, XGBoost `0.3363`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `101.0`, LSTM `0.4924`, XGBoost `0.3383`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `80.5`, LSTM `0.4959`, XGBoost `0.3441`, closer `xgboost`, smoke `3`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `101.5`, LSTM `0.4725`, XGBoost `0.3439`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `79.0`, LSTM `0.4797`, XGBoost `0.3540`, closer `xgboost`, smoke `3`, inferno `3`, utility_damage `0.0`, recent_utility `0`
- seconds `102.0`, LSTM `0.4604`, XGBoost `0.3439`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `80.0`, LSTM `0.4697`, XGBoost `0.3544`, closer `xgboost`, smoke `3`, inferno `3`, utility_damage `0.0`, recent_utility `0`
- seconds `78.0`, LSTM `0.5426`, XGBoost `0.6571`, closer `lstm`, smoke `3`, inferno `3`, utility_damage `0.0`, recent_utility `0`
- seconds `77.5`, LSTM `0.5430`, XGBoost `0.6571`, closer `lstm`, smoke `3`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `99.5`, LSTM `0.5188`, XGBoost `0.4073`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
