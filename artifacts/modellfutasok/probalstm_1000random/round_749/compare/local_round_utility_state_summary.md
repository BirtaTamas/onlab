# Local Round Utility State Analysis

- csv_path: `processed_full/blast_bounty_season_2/blast-bounty-2025-season-2-passion-ua-vs-spirit-bo3-WimU0hRkNcqhh3KAjCozBx/passion-ua-vs-spirit-m3-ancient.csv`
- round_num: `7`
- rows: `178`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 178 | 1.000 | 0.714299 | 0.715406 | -0.001107 | 115 | 63 | 1.000000 | 1.000000 |
| active/recent utility | 178 | 1.000 | 0.714299 | 0.715406 | -0.001107 | 115 | 63 | 1.000000 | 1.000000 |
| strong utility action | 157 | 0.882 | 0.715422 | 0.721052 | -0.005629 | 100 | 57 | 1.000000 | 1.000000 |
| utility damage | 11 | 0.062 | 0.577360 | 0.539537 | 0.037823 | 11 | 0 | 1.000000 | 1.000000 |
| active smoke/inferno | 157 | 0.882 | 0.715422 | 0.721052 | -0.005629 | 100 | 57 | 1.000000 | 1.000000 |
| recent utility last 5s | 10 | 0.056 | 0.794673 | 0.754695 | 0.039979 | 10 | 0 | 1.000000 | 1.000000 |
| flash effect present | 178 | 1.000 | 0.714299 | 0.715406 | -0.001107 | 115 | 63 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `7.0s` - `63.0s`, rows `113`
- `64.0s` - `85.5s`, rows `44`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `72.0`, LSTM `0.5318`, XGBoost `0.7101`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `70.5`, LSTM `0.5325`, XGBoost `0.7101`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `71.5`, LSTM `0.5374`, XGBoost `0.7101`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `70.0`, LSTM `0.5357`, XGBoost `0.7080`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `71.0`, LSTM `0.5382`, XGBoost `0.7101`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `69.5`, LSTM `0.5433`, XGBoost `0.7080`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `69.0`, LSTM `0.5622`, XGBoost `0.7080`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `79.5`, LSTM `0.6835`, XGBoost `0.8258`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `79.0`, LSTM `0.6921`, XGBoost `0.8256`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `81.0`, LSTM `0.6924`, XGBoost `0.8258`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
