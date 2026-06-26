# Local Round Utility State Analysis

- csv_path: `processed_full/blast_open_london/blast-open-london-2025-spirit-vs-furia-bo3-_zQK5XUu10iN1JLmPA8zQ4/spirit-vs-furia-m2-nuke.csv`
- round_num: `6`
- rows: `230`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 230 | 1.000 | 0.725829 | 0.784434 | -0.058605 | 49 | 181 | 1.000000 | 1.000000 |
| active/recent utility | 230 | 1.000 | 0.725829 | 0.784434 | -0.058605 | 49 | 181 | 1.000000 | 1.000000 |
| strong utility action | 121 | 0.526 | 0.674826 | 0.698620 | -0.023793 | 48 | 73 | 1.000000 | 1.000000 |
| utility damage | 11 | 0.048 | 0.676959 | 0.627074 | 0.049885 | 11 | 0 | 1.000000 | 1.000000 |
| active smoke/inferno | 106 | 0.461 | 0.679231 | 0.701049 | -0.021818 | 40 | 66 | 1.000000 | 1.000000 |
| recent utility last 5s | 22 | 0.096 | 0.650872 | 0.665542 | -0.014670 | 15 | 7 | 1.000000 | 1.000000 |
| flash effect present | 230 | 1.000 | 0.725829 | 0.784434 | -0.058605 | 49 | 181 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `8.5s` - `61.0s`, rows `106`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `1.5`, LSTM `0.6294`, XGBoost `0.7740`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `3`
- seconds `2.0`, LSTM `0.6307`, XGBoost `0.7724`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `3`
- seconds `2.5`, LSTM `0.6368`, XGBoost `0.7698`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `3`
- seconds `3.0`, LSTM `0.6383`, XGBoost `0.7687`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `3`
- seconds `61.0`, LSTM `0.7175`, XGBoost `0.8339`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `40.0`, LSTM `0.5530`, XGBoost `0.6694`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `60.5`, LSTM `0.7242`, XGBoost `0.8321`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `38.5`, LSTM `0.5573`, XGBoost `0.6637`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `60.0`, LSTM `0.7261`, XGBoost `0.8318`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `22.5`, LSTM `0.7522`, XGBoost `0.6468`, closer `lstm`, smoke `6`, inferno `0`, utility_damage `0.0`, recent_utility `0`
