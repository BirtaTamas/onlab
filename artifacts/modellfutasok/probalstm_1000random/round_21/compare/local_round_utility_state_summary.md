# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-wildcard-vs-legacy-bo3-NvI4DRplwm0O-zy6YVkFbj/wildcard-vs-legacy-m2-nuke.csv`
- round_num: `8`
- rows: `215`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 215 | 1.000 | 0.697894 | 0.738504 | -0.040610 | 62 | 153 | 1.000000 | 1.000000 |
| active/recent utility | 215 | 1.000 | 0.697894 | 0.738504 | -0.040610 | 62 | 153 | 1.000000 | 1.000000 |
| strong utility action | 188 | 0.874 | 0.701814 | 0.743292 | -0.041479 | 55 | 133 | 1.000000 | 1.000000 |
| utility damage | 10 | 0.047 | 0.660872 | 0.616376 | 0.044496 | 10 | 0 | 1.000000 | 1.000000 |
| active smoke/inferno | 188 | 0.874 | 0.701814 | 0.743292 | -0.041479 | 55 | 133 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 215 | 1.000 | 0.697894 | 0.738504 | -0.040610 | 62 | 153 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `8.0s` - `57.0s`, rows `99`
- `63.0s` - `107.0s`, rows `89`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `53.5`, LSTM `0.6705`, XGBoost `0.8267`, closer `xgboost`, smoke `1`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `79.0`, LSTM `0.6642`, XGBoost `0.8125`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `79.5`, LSTM `0.6655`, XGBoost `0.8125`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `78.5`, LSTM `0.6657`, XGBoost `0.8125`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `50.0`, LSTM `0.5981`, XGBoost `0.7441`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `80.0`, LSTM `0.6684`, XGBoost `0.8125`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `80.5`, LSTM `0.6709`, XGBoost `0.8125`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `81.5`, LSTM `0.6711`, XGBoost `0.8125`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `81.0`, LSTM `0.6712`, XGBoost `0.8125`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `50.5`, LSTM `0.6037`, XGBoost `0.7444`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
