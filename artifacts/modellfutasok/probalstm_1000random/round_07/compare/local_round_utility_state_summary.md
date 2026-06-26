# Local Round Utility State Analysis

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-lynn-vision-vs-3dmax-bo3-peFr0yEP4eKTMrYfeYqBZK/lynn-vision-vs-3dmax-m3-train.csv`
- round_num: `17`
- rows: `218`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 218 | 1.000 | 0.497989 | 0.552190 | -0.054201 | 3 | 215 | 0.261468 | 0.949541 |
| active/recent utility | 218 | 1.000 | 0.497989 | 0.552190 | -0.054201 | 3 | 215 | 0.261468 | 0.949541 |
| strong utility action | 195 | 0.894 | 0.501283 | 0.553935 | -0.052652 | 3 | 192 | 0.276923 | 0.943590 |
| utility damage | 10 | 0.046 | 0.918135 | 0.950641 | -0.032506 | 1 | 9 | 1.000000 | 1.000000 |
| active smoke/inferno | 185 | 0.849 | 0.501705 | 0.553953 | -0.052248 | 3 | 182 | 0.270270 | 0.940541 |
| recent utility last 5s | 10 | 0.046 | 0.493475 | 0.553609 | -0.060135 | 0 | 10 | 0.400000 | 1.000000 |
| flash effect present | 218 | 1.000 | 0.497989 | 0.552190 | -0.054201 | 3 | 215 | 0.261468 | 0.949541 |

## Active Smoke/Inferno Intervals

- `7.0s` - `66.0s`, rows `119`
- `76.0s` - `108.5s`, rows `66`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `91.0`, LSTM `0.5933`, XGBoost `0.7757`, closer `xgboost`, smoke `3`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `61.0`, LSTM `0.4264`, XGBoost `0.5289`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `98.5`, LSTM `0.4605`, XGBoost `0.5582`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `91.5`, LSTM `0.6796`, XGBoost `0.7768`, closer `xgboost`, smoke `3`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `60.5`, LSTM `0.4311`, XGBoost `0.5267`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `61.5`, LSTM `0.4370`, XGBoost `0.5289`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `58.0`, LSTM `0.4376`, XGBoost `0.5272`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `60.0`, LSTM `0.4395`, XGBoost `0.5270`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `59.5`, LSTM `0.4407`, XGBoost `0.5272`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `58.5`, LSTM `0.4434`, XGBoost `0.5272`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
