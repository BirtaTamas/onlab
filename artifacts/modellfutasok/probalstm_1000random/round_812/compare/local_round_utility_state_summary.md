# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-eternal-fire-vs-flyquest-bo3-bOv4otMGdpLsO1VdhzI_AV/eternal-fire-vs-flyquest-m2-nuke.csv`
- round_num: `2`
- rows: `165`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 165 | 1.000 | 0.298520 | 0.256279 | 0.042241 | 82 | 83 | 0.806061 | 1.000000 |
| active/recent utility | 165 | 1.000 | 0.298520 | 0.256279 | 0.042241 | 82 | 83 | 0.806061 | 1.000000 |
| strong utility action | 126 | 0.764 | 0.273350 | 0.232770 | 0.040579 | 63 | 63 | 0.817460 | 1.000000 |
| utility damage | 10 | 0.061 | 0.428406 | 0.344936 | 0.083469 | 3 | 7 | 0.800000 | 1.000000 |
| active smoke/inferno | 123 | 0.745 | 0.267915 | 0.230018 | 0.037896 | 63 | 60 | 0.813008 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 165 | 1.000 | 0.298520 | 0.256279 | 0.042241 | 82 | 83 | 0.806061 | 1.000000 |

## Active Smoke/Inferno Intervals

- `10.0s` - `35.0s`, rows `51`
- `46.5s` - `82.0s`, rows `72`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `27.5`, LSTM `0.5193`, XGBoost `0.3507`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `31.5`, LSTM `0.5183`, XGBoost `0.3525`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `47.0`, LSTM `0.5173`, XGBoost `0.3519`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `1.0`, recent_utility `0`
- seconds `32.0`, LSTM `0.5175`, XGBoost `0.3527`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `27.0`, LSTM `0.5119`, XGBoost `0.3476`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `32.5`, LSTM `0.5169`, XGBoost `0.3527`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `47.5`, LSTM `0.5155`, XGBoost `0.3519`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `1.0`, recent_utility `0`
- seconds `33.0`, LSTM `0.5156`, XGBoost `0.3521`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `30.5`, LSTM `0.5156`, XGBoost `0.3525`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `31.0`, LSTM `0.5154`, XGBoost `0.3525`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
