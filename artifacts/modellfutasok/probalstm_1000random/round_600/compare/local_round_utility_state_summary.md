# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-heroic-vs-3dmax-bo3-Dgk7HiwYvj5CMwMpEHLxHJ/heroic-vs-3dmax-m1-nuke.csv`
- round_num: `6`
- rows: `235`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 235 | 1.000 | 0.741724 | 0.762688 | -0.020964 | 99 | 136 | 0.927660 | 0.927660 |
| active/recent utility | 235 | 1.000 | 0.741724 | 0.762688 | -0.020964 | 99 | 136 | 0.927660 | 0.927660 |
| strong utility action | 185 | 0.787 | 0.775396 | 0.782954 | -0.007558 | 80 | 105 | 0.962162 | 0.962162 |
| utility damage | 21 | 0.089 | 0.728595 | 0.717332 | 0.011264 | 14 | 7 | 1.000000 | 1.000000 |
| active smoke/inferno | 178 | 0.757 | 0.777507 | 0.785994 | -0.008487 | 73 | 105 | 0.960674 | 0.960674 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 235 | 1.000 | 0.741724 | 0.762688 | -0.020964 | 99 | 136 | 0.927660 | 0.927660 |

## Active Smoke/Inferno Intervals

- `8.0s` - `51.5s`, rows `88`
- `53.5s` - `92.5s`, rows `79`
- `97.0s` - `102.0s`, rows `11`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `78.5`, LSTM `0.6455`, XGBoost `0.7686`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `79.0`, LSTM `0.6538`, XGBoost `0.7681`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `78.0`, LSTM `0.6578`, XGBoost `0.7686`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `73.0`, LSTM `0.7437`, XGBoost `0.8345`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `72.5`, LSTM `0.7423`, XGBoost `0.8330`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `72.0`, LSTM `0.7470`, XGBoost `0.8330`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `12.5`, LSTM `0.6058`, XGBoost `0.5324`, closer `lstm`, smoke `4`, inferno `2`, utility_damage `35.0`, recent_utility `0`
- seconds `19.5`, LSTM `0.9155`, XGBoost `0.9881`, closer `xgboost`, smoke `4`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `21.5`, LSTM `0.8708`, XGBoost `0.9415`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `11.5`, LSTM `0.6145`, XGBoost `0.5439`, closer `lstm`, smoke `4`, inferno `1`, utility_damage `35.0`, recent_utility `0`
