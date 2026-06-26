# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-heroic-vs-3dmax-bo3-OVT4ch_FfOW2E26liKqT_k/heroic-vs-3dmax-m2-inferno.csv`
- round_num: `7`
- rows: `140`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 140 | 1.000 | 0.656480 | 0.668053 | -0.011572 | 77 | 63 | 0.821429 | 0.771429 |
| active/recent utility | 140 | 1.000 | 0.656480 | 0.668053 | -0.011572 | 77 | 63 | 0.821429 | 0.771429 |
| strong utility action | 94 | 0.671 | 0.616010 | 0.638867 | -0.022857 | 56 | 38 | 0.734043 | 0.659574 |
| utility damage | 20 | 0.143 | 0.572202 | 0.511211 | 0.060991 | 19 | 1 | 0.700000 | 0.550000 |
| active smoke/inferno | 94 | 0.671 | 0.616010 | 0.638867 | -0.022857 | 56 | 38 | 0.734043 | 0.659574 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 140 | 1.000 | 0.656480 | 0.668053 | -0.011572 | 77 | 63 | 0.821429 | 0.771429 |

## Active Smoke/Inferno Intervals

- `10.5s` - `57.0s`, rows `94`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `50.0`, LSTM `0.7155`, XGBoost `0.8873`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `45.0`, LSTM `0.7123`, XGBoost `0.8837`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `49.5`, LSTM `0.7176`, XGBoost `0.8871`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `41.5`, LSTM `0.7050`, XGBoost `0.8733`, closer `xgboost`, smoke `3`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `49.0`, LSTM `0.7208`, XGBoost `0.8880`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `46.0`, LSTM `0.7213`, XGBoost `0.8845`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `52.0`, LSTM `0.7263`, XGBoost `0.8890`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `51.0`, LSTM `0.7265`, XGBoost `0.8886`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `51.5`, LSTM `0.7288`, XGBoost `0.8890`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `44.0`, LSTM `0.7188`, XGBoost `0.8788`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
