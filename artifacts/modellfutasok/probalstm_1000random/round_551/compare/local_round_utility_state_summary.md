# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-liquid-vs-3dmax-bo3-k7r_vGkiL4eRhxKdRPUZx1/liquid-vs-3dmax-m2-ancient.csv`
- round_num: `18`
- rows: `113`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 113 | 1.000 | 0.206671 | 0.188452 | 0.018219 | 73 | 40 | 0.734513 | 0.734513 |
| active/recent utility | 113 | 1.000 | 0.206671 | 0.188452 | 0.018219 | 73 | 40 | 0.734513 | 0.734513 |
| strong utility action | 70 | 0.619 | 0.220499 | 0.202811 | 0.017688 | 43 | 27 | 0.757143 | 0.757143 |
| utility damage | 12 | 0.106 | 0.555156 | 0.544593 | 0.010563 | 4 | 8 | 0.416667 | 0.416667 |
| active smoke/inferno | 70 | 0.619 | 0.220499 | 0.202811 | 0.017688 | 43 | 27 | 0.757143 | 0.757143 |
| recent utility last 5s | 10 | 0.088 | 0.001497 | 0.006608 | -0.005111 | 10 | 0 | 1.000000 | 1.000000 |
| flash effect present | 113 | 1.000 | 0.206671 | 0.188452 | 0.018219 | 73 | 40 | 0.734513 | 0.734513 |

## Active Smoke/Inferno Intervals

- `6.5s` - `41.0s`, rows `70`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `18.5`, LSTM `0.4756`, XGBoost `0.3010`, closer `xgboost`, smoke `5`, inferno `2`, utility_damage `56.0`, recent_utility `0`
- seconds `19.5`, LSTM `0.4734`, XGBoost `0.3000`, closer `xgboost`, smoke `6`, inferno `2`, utility_damage `56.0`, recent_utility `0`
- seconds `19.0`, LSTM `0.4602`, XGBoost `0.3000`, closer `xgboost`, smoke `5`, inferno `2`, utility_damage `56.0`, recent_utility `0`
- seconds `18.0`, LSTM `0.4528`, XGBoost `0.3010`, closer `xgboost`, smoke `5`, inferno `2`, utility_damage `64.0`, recent_utility `0`
- seconds `17.5`, LSTM `0.4325`, XGBoost `0.3010`, closer `xgboost`, smoke `5`, inferno `2`, utility_damage `64.0`, recent_utility `0`
- seconds `7.0`, LSTM `0.6074`, XGBoost `0.5265`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `6.5`, LSTM `0.5982`, XGBoost `0.5265`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `9.0`, LSTM `0.6019`, XGBoost `0.5363`, closer `xgboost`, smoke `3`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `8.5`, LSTM `0.5972`, XGBoost `0.5331`, closer `xgboost`, smoke `3`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `8.0`, LSTM `0.5991`, XGBoost `0.5355`, closer `xgboost`, smoke `3`, inferno `2`, utility_damage `0.0`, recent_utility `0`
