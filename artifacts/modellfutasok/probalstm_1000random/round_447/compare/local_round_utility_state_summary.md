# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-eternal-fire-vs-flyquest-bo3-bOv4otMGdpLsO1VdhzI_AV/eternal-fire-vs-flyquest-m2-nuke.csv`
- round_num: `5`
- rows: `208`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 208 | 1.000 | 0.218003 | 0.231975 | -0.013972 | 124 | 84 | 1.000000 | 1.000000 |
| active/recent utility | 208 | 1.000 | 0.218003 | 0.231975 | -0.013972 | 124 | 84 | 1.000000 | 1.000000 |
| strong utility action | 158 | 0.760 | 0.242906 | 0.243851 | -0.000945 | 86 | 72 | 1.000000 | 1.000000 |
| utility damage | 11 | 0.053 | 0.027712 | 0.088383 | -0.060671 | 11 | 0 | 1.000000 | 1.000000 |
| active smoke/inferno | 148 | 0.712 | 0.240446 | 0.239460 | 0.000986 | 76 | 72 | 1.000000 | 1.000000 |
| recent utility last 5s | 10 | 0.048 | 0.279309 | 0.308832 | -0.029523 | 10 | 0 | 1.000000 | 1.000000 |
| flash effect present | 208 | 1.000 | 0.218003 | 0.231975 | -0.013972 | 124 | 84 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `9.0s` - `34.5s`, rows `52`
- `37.0s` - `58.5s`, rows `44`
- `66.0s` - `91.5s`, rows `52`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `80.5`, LSTM `0.0465`, XGBoost `0.2846`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `80.0`, LSTM `0.0512`, XGBoost `0.2828`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `81.0`, LSTM `0.0522`, XGBoost `0.2815`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `30.0`, LSTM `0.4656`, XGBoost `0.2834`, closer `xgboost`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `29.0`, LSTM `0.4627`, XGBoost `0.2834`, closer `xgboost`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `31.5`, LSTM `0.4625`, XGBoost `0.2839`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `30.5`, LSTM `0.4618`, XGBoost `0.2837`, closer `xgboost`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `29.5`, LSTM `0.4586`, XGBoost `0.2834`, closer `xgboost`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `32.5`, LSTM `0.4578`, XGBoost `0.2873`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `22.5`, LSTM `0.4729`, XGBoost `0.3029`, closer `xgboost`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
