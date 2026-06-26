# Local Round Utility State Analysis

- csv_path: `processed_full/blast_bounty_season_1/blast-bounty-2025-season-1-eternal-fire-vs-falcons-bo3-Bm3FkXiO5h_cvpKxUnOmaW/eternal-fire-vs-falcons-m1-inferno.csv`
- round_num: `16`
- rows: `193`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 193 | 1.000 | 0.918056 | 0.978307 | -0.060251 | 0 | 193 | 1.000000 | 1.000000 |
| active/recent utility | 193 | 1.000 | 0.918056 | 0.978307 | -0.060251 | 0 | 193 | 1.000000 | 1.000000 |
| strong utility action | 149 | 0.772 | 0.910836 | 0.976777 | -0.065941 | 0 | 149 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 149 | 0.772 | 0.910836 | 0.976777 | -0.065941 | 0 | 149 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 193 | 1.000 | 0.918056 | 0.978307 | -0.060251 | 0 | 193 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `8.5s` - `37.0s`, rows `58`
- `40.5s` - `85.5s`, rows `91`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `61.5`, LSTM `0.8774`, XGBoost `0.9780`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `62.5`, LSTM `0.8790`, XGBoost `0.9778`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `62.0`, LSTM `0.8804`, XGBoost `0.9778`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `61.0`, LSTM `0.8820`, XGBoost `0.9780`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `64.0`, LSTM `0.8843`, XGBoost `0.9778`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `60.5`, LSTM `0.8846`, XGBoost `0.9779`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `63.5`, LSTM `0.8851`, XGBoost `0.9778`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `60.0`, LSTM `0.8861`, XGBoost `0.9779`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `58.5`, LSTM `0.8854`, XGBoost `0.9759`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `52.0`, LSTM `0.8863`, XGBoost `0.9760`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
