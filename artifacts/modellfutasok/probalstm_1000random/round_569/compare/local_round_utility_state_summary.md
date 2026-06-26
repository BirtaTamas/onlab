# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-eternal-fire-vs-flyquest-bo3-bOv4otMGdpLsO1VdhzI_AV/eternal-fire-vs-flyquest-m2-nuke.csv`
- round_num: `7`
- rows: `154`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 154 | 1.000 | 0.306937 | 0.331016 | -0.024079 | 118 | 36 | 0.909091 | 0.610390 |
| active/recent utility | 154 | 1.000 | 0.306937 | 0.331016 | -0.024079 | 118 | 36 | 0.909091 | 0.610390 |
| strong utility action | 135 | 0.877 | 0.279777 | 0.300227 | -0.020450 | 99 | 36 | 0.970370 | 0.696296 |
| utility damage | 20 | 0.130 | 0.458709 | 0.519391 | -0.060682 | 19 | 1 | 0.850000 | 0.100000 |
| active smoke/inferno | 135 | 0.877 | 0.279777 | 0.300227 | -0.020450 | 99 | 36 | 0.970370 | 0.696296 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 154 | 1.000 | 0.306937 | 0.331016 | -0.024079 | 118 | 36 | 0.909091 | 0.610390 |

## Active Smoke/Inferno Intervals

- `9.5s` - `76.5s`, rows `135`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `50.5`, LSTM `0.1873`, XGBoost `0.3064`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `50.0`, LSTM `0.1960`, XGBoost `0.3077`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `49.5`, LSTM `0.2082`, XGBoost `0.3077`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `25.0`, LSTM `0.3926`, XGBoost `0.3041`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `55.0`, LSTM `0.4245`, XGBoost `0.5108`, closer `lstm`, smoke `3`, inferno `2`, utility_damage `56.0`, recent_utility `0`
- seconds `48.0`, LSTM `0.2229`, XGBoost `0.3085`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `43.5`, LSTM `0.3936`, XGBoost `0.3100`, closer `xgboost`, smoke `4`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `13.0`, LSTM `0.4725`, XGBoost `0.5550`, closer `lstm`, smoke `0`, inferno `3`, utility_damage `3.0`, recent_utility `0`
- seconds `12.5`, LSTM `0.4746`, XGBoost `0.5550`, closer `lstm`, smoke `0`, inferno `3`, utility_damage `3.0`, recent_utility `0`
- seconds `23.0`, LSTM `0.4743`, XGBoost `0.5545`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
