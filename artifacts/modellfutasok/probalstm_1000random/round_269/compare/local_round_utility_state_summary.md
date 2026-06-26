# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-astralis-vs-gamerlegion-bo3-8K-MOEPC1meC7FXyBc8fA2/astralis-vs-gamerlegion-m1-nuke.csv`
- round_num: `10`
- rows: `179`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 179 | 1.000 | 0.556410 | 0.556598 | -0.000188 | 78 | 101 | 0.709497 | 0.681564 |
| active/recent utility | 179 | 1.000 | 0.556410 | 0.556598 | -0.000188 | 78 | 101 | 0.709497 | 0.681564 |
| strong utility action | 138 | 0.771 | 0.485312 | 0.491804 | -0.006492 | 57 | 81 | 0.623188 | 0.586957 |
| utility damage | 20 | 0.112 | 0.529147 | 0.556370 | -0.027223 | 3 | 17 | 0.850000 | 0.800000 |
| active smoke/inferno | 138 | 0.771 | 0.485312 | 0.491804 | -0.006492 | 57 | 81 | 0.623188 | 0.586957 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 179 | 1.000 | 0.556410 | 0.556598 | -0.000188 | 78 | 101 | 0.709497 | 0.681564 |

## Active Smoke/Inferno Intervals

- `8.0s` - `71.0s`, rows `127`
- `72.5s` - `77.5s`, rows `11`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `45.5`, LSTM `0.2904`, XGBoost `0.4852`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `46.0`, LSTM `0.2998`, XGBoost `0.4867`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `73.0`, LSTM `0.7699`, XGBoost `0.5847`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `73.5`, LSTM `0.7625`, XGBoost `0.5847`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `72.5`, LSTM `0.7574`, XGBoost `0.5847`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `47.0`, LSTM `0.3210`, XGBoost `0.4861`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `74.0`, LSTM `0.7461`, XGBoost `0.5847`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `46.5`, LSTM `0.3248`, XGBoost `0.4861`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `47.5`, LSTM `0.3292`, XGBoost `0.4861`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `75.5`, LSTM `0.7389`, XGBoost `0.5847`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
