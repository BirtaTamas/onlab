# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-astralis-vs-ence-bo3-avzZyhQ46OR9GyYE_6NeM7/astralis-vs-ence-m2-nuke.csv`
- round_num: `10`
- rows: `189`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 189 | 1.000 | 0.682338 | 0.714383 | -0.032045 | 32 | 157 | 0.989418 | 0.989418 |
| active/recent utility | 189 | 1.000 | 0.682338 | 0.714383 | -0.032045 | 32 | 157 | 0.989418 | 0.989418 |
| strong utility action | 66 | 0.349 | 0.569765 | 0.587788 | -0.018022 | 17 | 49 | 0.969697 | 0.969697 |
| utility damage | 10 | 0.053 | 0.582275 | 0.581220 | 0.001055 | 4 | 6 | 1.000000 | 1.000000 |
| active smoke/inferno | 66 | 0.349 | 0.569765 | 0.587788 | -0.018022 | 17 | 49 | 0.969697 | 0.969697 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 189 | 1.000 | 0.682338 | 0.714383 | -0.032045 | 32 | 157 | 0.989418 | 0.989418 |

## Active Smoke/Inferno Intervals

- `7.5s` - `40.0s`, rows `66`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `38.5`, LSTM `0.5379`, XGBoost `0.6094`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `38.0`, LSTM `0.5405`, XGBoost `0.6094`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `36.5`, LSTM `0.5418`, XGBoost `0.6094`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `37.5`, LSTM `0.5423`, XGBoost `0.6094`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `39.0`, LSTM `0.5439`, XGBoost `0.6094`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `37.0`, LSTM `0.5462`, XGBoost `0.6094`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `39.5`, LSTM `0.5483`, XGBoost `0.6094`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `40.0`, LSTM `0.5479`, XGBoost `0.6086`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `30.5`, LSTM `0.5613`, XGBoost `0.6218`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `36.0`, LSTM `0.5505`, XGBoost `0.6094`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
