# Local Round Utility State Analysis

- csv_path: `processed_full/blast_bounty_season_1_finals/blast-bounty-2025-season-1-finals-g2-vs-betboom-bo3-pCfbtiY01aL_JW2Hy1pnZ6/g2-vs-betboom-m1-anubis.csv`
- round_num: `10`
- rows: `171`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 171 | 1.000 | 0.247209 | 0.237928 | 0.009281 | 96 | 75 | 0.906433 | 1.000000 |
| active/recent utility | 171 | 1.000 | 0.247209 | 0.237928 | 0.009281 | 96 | 75 | 0.906433 | 1.000000 |
| strong utility action | 111 | 0.649 | 0.314839 | 0.298149 | 0.016691 | 45 | 66 | 0.855856 | 1.000000 |
| utility damage | 12 | 0.070 | 0.344710 | 0.330252 | 0.014457 | 3 | 9 | 1.000000 | 1.000000 |
| active smoke/inferno | 111 | 0.649 | 0.314839 | 0.298149 | 0.016691 | 45 | 66 | 0.855856 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 171 | 1.000 | 0.247209 | 0.237928 | 0.009281 | 96 | 75 | 0.906433 | 1.000000 |

## Active Smoke/Inferno Intervals

- `7.0s` - `56.5s`, rows `100`
- `69.5s` - `74.5s`, rows `11`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `47.0`, LSTM `0.5512`, XGBoost `0.4190`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `8.0`, recent_utility `0`
- seconds `47.5`, LSTM `0.5555`, XGBoost `0.4239`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `8.0`, recent_utility `0`
- seconds `48.5`, LSTM `0.5360`, XGBoost `0.4138`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `8.0`, recent_utility `0`
- seconds `48.0`, LSTM `0.5347`, XGBoost `0.4143`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `8.0`, recent_utility `0`
- seconds `45.0`, LSTM `0.5616`, XGBoost `0.4419`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `14.0`, recent_utility `0`
- seconds `37.0`, LSTM `0.2778`, XGBoost `0.1594`, closer `xgboost`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `45.5`, LSTM `0.5600`, XGBoost `0.4419`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `13.0`, recent_utility `0`
- seconds `44.5`, LSTM `0.5616`, XGBoost `0.4447`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `14.0`, recent_utility `0`
- seconds `49.0`, LSTM `0.5213`, XGBoost `0.4131`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `8.0`, recent_utility `0`
- seconds `46.5`, LSTM `0.5503`, XGBoost `0.4462`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
