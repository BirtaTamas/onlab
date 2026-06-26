# Local Round Utility State Analysis

- csv_path: `processed_full/blast_bounty_season_2_finals/blast-bounty-2025-season-2-finals-the-mongolz-vs-aurora-bo3-BL3Rcvf733R8cy7TtLY5HE/the-mongolz-vs-aurora-m1-dust2.csv`
- round_num: `11`
- rows: `158`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 158 | 1.000 | 0.779050 | 0.786861 | -0.007811 | 66 | 92 | 1.000000 | 1.000000 |
| active/recent utility | 158 | 1.000 | 0.779050 | 0.786861 | -0.007811 | 66 | 92 | 1.000000 | 1.000000 |
| strong utility action | 127 | 0.804 | 0.793488 | 0.799891 | -0.006403 | 57 | 70 | 1.000000 | 1.000000 |
| utility damage | 20 | 0.127 | 0.966691 | 0.990674 | -0.023984 | 0 | 20 | 1.000000 | 1.000000 |
| active smoke/inferno | 111 | 0.703 | 0.801710 | 0.808986 | -0.007275 | 50 | 61 | 1.000000 | 1.000000 |
| recent utility last 5s | 16 | 0.101 | 0.736448 | 0.736800 | -0.000352 | 7 | 9 | 1.000000 | 1.000000 |
| flash effect present | 158 | 1.000 | 0.779050 | 0.786861 | -0.007811 | 66 | 92 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `10.5s` - `40.0s`, rows `60`
- `53.5s` - `78.5s`, rows `51`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `59.0`, LSTM `0.5740`, XGBoost `0.6774`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `60.0`, LSTM `0.6462`, XGBoost `0.7396`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `61.0`, LSTM `0.9137`, XGBoost `0.9876`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `60.5`, LSTM `0.6693`, XGBoost `0.7399`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `58.5`, LSTM `0.6145`, XGBoost `0.6774`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `59.5`, LSTM `0.5901`, XGBoost `0.6457`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `58.0`, LSTM `0.6256`, XGBoost `0.6774`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `61.5`, LSTM `0.9410`, XGBoost `0.9876`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `62.0`, LSTM `0.9492`, XGBoost `0.9877`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `35.5`, LSTM `0.7685`, XGBoost `0.7338`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
