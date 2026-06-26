# Local Round Utility State Analysis

- csv_path: `processed_full/iem_dallas/iem-dallas-2025-nrg-vs-aurora-bo3-qymu5EnF_DYwHSVf1aSLaG/nrg-vs-aurora-m1-inferno.csv`
- round_num: `15`
- rows: `213`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 213 | 1.000 | 0.127952 | 0.100646 | 0.027306 | 145 | 68 | 0.985915 | 1.000000 |
| active/recent utility | 213 | 1.000 | 0.127952 | 0.100646 | 0.027306 | 145 | 68 | 0.985915 | 1.000000 |
| strong utility action | 195 | 0.915 | 0.137176 | 0.105470 | 0.031705 | 127 | 68 | 0.984615 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 185 | 0.869 | 0.140701 | 0.106351 | 0.034350 | 120 | 65 | 0.983784 | 1.000000 |
| recent utility last 5s | 10 | 0.047 | 0.071962 | 0.089184 | -0.017222 | 7 | 3 | 1.000000 | 1.000000 |
| flash effect present | 213 | 1.000 | 0.127952 | 0.100646 | 0.027306 | 145 | 68 | 0.985915 | 1.000000 |

## Active Smoke/Inferno Intervals

- `9.5s` - `101.5s`, rows `185`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `54.5`, LSTM `0.5102`, XGBoost `0.2027`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `54.0`, LSTM `0.5094`, XGBoost `0.2027`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `53.5`, LSTM `0.5028`, XGBoost `0.2027`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `53.0`, LSTM `0.4997`, XGBoost `0.2027`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `52.5`, LSTM `0.4963`, XGBoost `0.2002`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `55.0`, LSTM `0.4968`, XGBoost `0.2025`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `55.5`, LSTM `0.4858`, XGBoost `0.2025`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `51.5`, LSTM `0.4743`, XGBoost `0.1996`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `52.0`, LSTM `0.4725`, XGBoost `0.1997`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `51.0`, LSTM `0.4672`, XGBoost `0.2014`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
