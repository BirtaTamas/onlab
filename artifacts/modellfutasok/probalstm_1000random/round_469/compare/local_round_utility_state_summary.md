# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-imperial-vs-metizport-bo3-yMtoBsoZq-jiQ0fSUscH7u/imperial-vs-metizport-m2-dust2.csv`
- round_num: `13`
- rows: `132`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 132 | 1.000 | 0.717071 | 0.729416 | -0.012345 | 48 | 84 | 1.000000 | 0.984848 |
| active/recent utility | 132 | 1.000 | 0.717071 | 0.729416 | -0.012345 | 48 | 84 | 1.000000 | 0.984848 |
| strong utility action | 97 | 0.735 | 0.728517 | 0.745032 | -0.016514 | 26 | 71 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 87 | 0.659 | 0.753776 | 0.772901 | -0.019125 | 16 | 71 | 1.000000 | 1.000000 |
| recent utility last 5s | 10 | 0.076 | 0.508767 | 0.502567 | 0.006200 | 10 | 0 | 1.000000 | 1.000000 |
| flash effect present | 132 | 1.000 | 0.717071 | 0.729416 | -0.012345 | 48 | 84 | 1.000000 | 0.984848 |

## Active Smoke/Inferno Intervals

- `16.0s` - `59.0s`, rows `87`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `35.5`, LSTM `0.7297`, XGBoost `0.8482`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `36.0`, LSTM `0.7839`, XGBoost `0.8490`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `37.0`, LSTM `0.7330`, XGBoost `0.7910`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `36.5`, LSTM `0.7789`, XGBoost `0.8359`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `33.0`, LSTM `0.5208`, XGBoost `0.5762`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `39.0`, LSTM `0.8797`, XGBoost `0.9327`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `33.5`, LSTM `0.5607`, XGBoost `0.6099`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `39.5`, LSTM `0.9447`, XGBoost `0.9934`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `35.0`, LSTM `0.5632`, XGBoost `0.6099`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `38.0`, LSTM `0.8739`, XGBoost `0.9175`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
