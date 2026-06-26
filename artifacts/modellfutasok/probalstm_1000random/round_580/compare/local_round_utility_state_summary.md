# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-astralis-vs-heroic-bo3-VpF2znQtwzecEgVsCr-4Wn/astralis-vs-heroic-m3-nuke.csv`
- round_num: `7`
- rows: `196`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 196 | 1.000 | 0.653345 | 0.722674 | -0.069329 | 34 | 162 | 0.908163 | 0.943878 |
| active/recent utility | 196 | 1.000 | 0.653345 | 0.722674 | -0.069329 | 34 | 162 | 0.908163 | 0.943878 |
| strong utility action | 135 | 0.689 | 0.640939 | 0.696479 | -0.055539 | 21 | 114 | 0.903704 | 0.918519 |
| utility damage | 25 | 0.128 | 0.550872 | 0.556411 | -0.005538 | 9 | 16 | 0.720000 | 0.680000 |
| active smoke/inferno | 135 | 0.689 | 0.640939 | 0.696479 | -0.055539 | 21 | 114 | 0.903704 | 0.918519 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 196 | 1.000 | 0.653345 | 0.722674 | -0.069329 | 34 | 162 | 0.908163 | 0.943878 |

## Active Smoke/Inferno Intervals

- `7.0s` - `74.0s`, rows `135`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `65.5`, LSTM `0.3406`, XGBoost `0.6270`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `72.5`, LSTM `0.5558`, XGBoost `0.8019`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `29.0`, recent_utility `0`
- seconds `68.0`, LSTM `0.5815`, XGBoost `0.8126`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `4.0`, recent_utility `0`
- seconds `68.5`, LSTM `0.5786`, XGBoost `0.8061`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `17.0`, recent_utility `0`
- seconds `74.0`, LSTM `0.5832`, XGBoost `0.8037`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `73.5`, LSTM `0.5847`, XGBoost `0.8037`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `12.0`, recent_utility `0`
- seconds `73.0`, LSTM `0.5866`, XGBoost `0.8037`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `25.0`, recent_utility `0`
- seconds `72.0`, LSTM `0.5892`, XGBoost `0.8019`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `29.0`, recent_utility `0`
- seconds `71.5`, LSTM `0.5898`, XGBoost `0.8019`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `29.0`, recent_utility `0`
- seconds `65.0`, LSTM `0.4254`, XGBoost `0.6221`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
