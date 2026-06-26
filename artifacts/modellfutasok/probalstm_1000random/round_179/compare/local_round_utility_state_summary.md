# Local Round Utility State Analysis

- csv_path: `processed_full/iem_cologne/iem-cologne-2025-the-mongolz-vs-3dmax-bo3-NhOpC3bR-AJd86c-60IeuJ/the-mongolz-vs-3dmax-m1-nuke.csv`
- round_num: `4`
- rows: `230`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 230 | 1.000 | 0.542402 | 0.595216 | -0.052814 | 7 | 223 | 0.643478 | 1.000000 |
| active/recent utility | 230 | 1.000 | 0.542402 | 0.595216 | -0.052814 | 7 | 223 | 0.643478 | 1.000000 |
| strong utility action | 166 | 0.722 | 0.483258 | 0.536134 | -0.052876 | 6 | 160 | 0.596386 | 1.000000 |
| utility damage | 13 | 0.057 | 0.496719 | 0.548550 | -0.051831 | 0 | 13 | 0.307692 | 1.000000 |
| active smoke/inferno | 166 | 0.722 | 0.483258 | 0.536134 | -0.052876 | 6 | 160 | 0.596386 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 230 | 1.000 | 0.542402 | 0.595216 | -0.052814 | 7 | 223 | 0.643478 | 1.000000 |

## Active Smoke/Inferno Intervals

- `9.0s` - `69.5s`, rows `122`
- `75.0s` - `96.5s`, rows `44`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `82.5`, LSTM `0.3250`, XGBoost `0.5186`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `75.0`, LSTM `0.3370`, XGBoost `0.5181`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `95.0`, LSTM `0.3343`, XGBoost `0.5107`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `96.0`, LSTM `0.3369`, XGBoost `0.5034`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `95.5`, LSTM `0.3380`, XGBoost `0.5028`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `82.0`, LSTM `0.3547`, XGBoost `0.5120`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `96.5`, LSTM `0.3530`, XGBoost `0.5028`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `84.5`, LSTM `0.3832`, XGBoost `0.5270`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `83.0`, LSTM `0.3857`, XGBoost `0.5247`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `79.5`, LSTM `0.3801`, XGBoost `0.5131`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
