# Local Round Utility State Analysis

- csv_path: `processed_full/blast_rivals_season_2/blast-rivals-2025-season-2-furia-vs-pain-bo3-BGpRMXEt8xpbRAS7KbpPH6/furia-vs-pain-m2-overpass.csv`
- round_num: `26`
- rows: `171`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 171 | 1.000 | 0.701714 | 0.703418 | -0.001704 | 66 | 105 | 0.795322 | 0.982456 |
| active/recent utility | 171 | 1.000 | 0.701714 | 0.703418 | -0.001704 | 66 | 105 | 0.795322 | 0.982456 |
| strong utility action | 143 | 0.836 | 0.696151 | 0.707107 | -0.010956 | 45 | 98 | 0.755245 | 0.979021 |
| utility damage | 10 | 0.058 | 0.608038 | 0.561904 | 0.046134 | 10 | 0 | 1.000000 | 1.000000 |
| active smoke/inferno | 143 | 0.836 | 0.696151 | 0.707107 | -0.010956 | 45 | 98 | 0.755245 | 0.979021 |
| recent utility last 5s | 10 | 0.058 | 0.604452 | 0.562435 | 0.042017 | 10 | 0 | 1.000000 | 1.000000 |
| flash effect present | 171 | 1.000 | 0.701714 | 0.703418 | -0.001704 | 66 | 105 | 0.795322 | 0.982456 |

## Active Smoke/Inferno Intervals

- `9.0s` - `55.0s`, rows `93`
- `57.0s` - `81.5s`, rows `50`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `48.5`, LSTM `0.5821`, XGBoost `0.7694`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `24.0`, LSTM `0.4851`, XGBoost `0.3201`, closer `lstm`, smoke `6`, inferno `1`, utility_damage `53.0`, recent_utility `0`
- seconds `24.5`, LSTM `0.4728`, XGBoost `0.3201`, closer `lstm`, smoke `6`, inferno `1`, utility_damage `53.0`, recent_utility `0`
- seconds `25.0`, LSTM `0.4637`, XGBoost `0.3201`, closer `lstm`, smoke `6`, inferno `1`, utility_damage `53.0`, recent_utility `0`
- seconds `49.0`, LSTM `0.6451`, XGBoost `0.7694`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `27.5`, LSTM `0.6585`, XGBoost `0.7638`, closer `xgboost`, smoke `6`, inferno `1`, utility_damage `40.0`, recent_utility `0`
- seconds `9.0`, LSTM `0.6356`, XGBoost `0.5369`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `49.5`, LSTM `0.6726`, XGBoost `0.7694`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `28.0`, LSTM `0.6763`, XGBoost `0.7638`, closer `xgboost`, smoke `6`, inferno `0`, utility_damage `24.0`, recent_utility `0`
- seconds `9.5`, LSTM `0.6246`, XGBoost `0.5377`, closer `lstm`, smoke `0`, inferno `3`, utility_damage `0.0`, recent_utility `0`
