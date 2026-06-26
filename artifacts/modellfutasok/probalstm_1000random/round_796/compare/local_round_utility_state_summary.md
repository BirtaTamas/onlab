# Local Round Utility State Analysis

- csv_path: `processed_full/blast_bounty_season_2_finals/blast-bounty-2025-season-2-finals-the-mongolz-vs-aurora-bo3-BL3Rcvf733R8cy7TtLY5HE/the-mongolz-vs-aurora-m1-dust2.csv`
- round_num: `8`
- rows: `133`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 133 | 1.000 | 0.757879 | 0.776998 | -0.019119 | 29 | 104 | 1.000000 | 1.000000 |
| active/recent utility | 133 | 1.000 | 0.757879 | 0.776998 | -0.019119 | 29 | 104 | 1.000000 | 1.000000 |
| strong utility action | 130 | 0.977 | 0.758183 | 0.777017 | -0.018834 | 29 | 101 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 112 | 0.842 | 0.764320 | 0.775368 | -0.011048 | 29 | 83 | 1.000000 | 1.000000 |
| recent utility last 5s | 18 | 0.135 | 0.720001 | 0.787279 | -0.067278 | 0 | 18 | 1.000000 | 1.000000 |
| flash effect present | 133 | 1.000 | 0.757879 | 0.776998 | -0.019119 | 29 | 104 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `10.5s` - `66.0s`, rows `112`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `4.0`, LSTM `0.7108`, XGBoost `0.8376`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `3`
- seconds `3.5`, LSTM `0.7153`, XGBoost `0.8396`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `3`
- seconds `3.0`, LSTM `0.7240`, XGBoost `0.8387`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `3`
- seconds `2.5`, LSTM `0.7341`, XGBoost `0.8387`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `3`
- seconds `2.0`, LSTM `0.7408`, XGBoost `0.8381`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `3`
- seconds `1.5`, LSTM `0.7506`, XGBoost `0.8384`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `3`
- seconds `5.0`, LSTM `0.6733`, XGBoost `0.7473`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `4`
- seconds `1.0`, LSTM `0.7767`, XGBoost `0.8466`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `2`
- seconds `4.5`, LSTM `0.6775`, XGBoost `0.7470`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `4`
- seconds `52.5`, LSTM `0.6921`, XGBoost `0.6272`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
