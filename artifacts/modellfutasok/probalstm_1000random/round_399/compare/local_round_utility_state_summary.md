# Local Round Utility State Analysis

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-3dmax-vs-faze-bo3-oPmZd_fSjJ93cG46OkNLxM/3dmax-vs-faze-m2-ancient.csv`
- round_num: `5`
- rows: `202`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 202 | 1.000 | 0.108313 | 0.107698 | 0.000615 | 149 | 53 | 0.851485 | 0.856436 |
| active/recent utility | 202 | 1.000 | 0.108313 | 0.107698 | 0.000615 | 149 | 53 | 0.851485 | 0.856436 |
| strong utility action | 93 | 0.460 | 0.160951 | 0.160623 | 0.000328 | 52 | 41 | 0.806452 | 0.817204 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 93 | 0.460 | 0.160951 | 0.160623 | 0.000328 | 52 | 41 | 0.806452 | 0.817204 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 202 | 1.000 | 0.108313 | 0.107698 | 0.000615 | 149 | 53 | 0.851485 | 0.856436 |

## Active Smoke/Inferno Intervals

- `6.0s` - `52.0s`, rows `93`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `24.0`, LSTM `0.1525`, XGBoost `0.0878`, closer `xgboost`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `23.5`, LSTM `0.1432`, XGBoost `0.0878`, closer `xgboost`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `24.5`, LSTM `0.1406`, XGBoost `0.0878`, closer `xgboost`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `23.0`, LSTM `0.1399`, XGBoost `0.0878`, closer `xgboost`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `25.0`, LSTM `0.1396`, XGBoost `0.0880`, closer `xgboost`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `17.0`, LSTM `0.2748`, XGBoost `0.2286`, closer `xgboost`, smoke `4`, inferno `2`, utility_damage `91.0`, recent_utility `0`
- seconds `22.5`, LSTM `0.1280`, XGBoost `0.0881`, closer `xgboost`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `22.0`, LSTM `0.1254`, XGBoost `0.0881`, closer `xgboost`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `7.5`, LSTM `0.5587`, XGBoost `0.5235`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `19.5`, LSTM `0.0550`, XGBoost `0.0846`, closer `lstm`, smoke `4`, inferno `2`, utility_damage `43.0`, recent_utility `0`
