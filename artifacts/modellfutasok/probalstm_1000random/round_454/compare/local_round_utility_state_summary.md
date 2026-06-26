# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major_stage_2/blasttv-austin-major-2025-stage-2-b8-vs-lynn-vision-bo3-Whl3pjYuIoHffY1VOn8vws/b8-vs-lynn-vision-m1-dust2.csv`
- round_num: `7`
- rows: `229`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 229 | 1.000 | 0.396267 | 0.522496 | -0.126229 | 226 | 3 | 0.956332 | 0.174672 |
| active/recent utility | 229 | 1.000 | 0.396267 | 0.522496 | -0.126229 | 226 | 3 | 0.956332 | 0.174672 |
| strong utility action | 222 | 0.969 | 0.394887 | 0.523270 | -0.128383 | 219 | 3 | 0.954955 | 0.153153 |
| utility damage | 35 | 0.153 | 0.394480 | 0.535284 | -0.140804 | 35 | 0 | 1.000000 | 0.057143 |
| active smoke/inferno | 212 | 0.926 | 0.392784 | 0.525010 | -0.132226 | 209 | 3 | 0.952830 | 0.113208 |
| recent utility last 5s | 10 | 0.044 | 0.439476 | 0.486388 | -0.046912 | 10 | 0 | 1.000000 | 1.000000 |
| flash effect present | 229 | 1.000 | 0.396267 | 0.522496 | -0.126229 | 226 | 3 | 0.956332 | 0.174672 |

## Active Smoke/Inferno Intervals

- `8.0s` - `77.5s`, rows `140`
- `78.5s` - `114.0s`, rows `72`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `96.5`, LSTM `0.2579`, XGBoost `0.5196`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `97.0`, LSTM `0.2650`, XGBoost `0.5192`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `97.5`, LSTM `0.2900`, XGBoost `0.4997`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `94.0`, LSTM `0.3424`, XGBoost `0.5500`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `89.5`, LSTM `0.3601`, XGBoost `0.5519`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `99.0`, LSTM `0.3147`, XGBoost `0.5037`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `98.0`, LSTM `0.3133`, XGBoost `0.5019`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `94.5`, LSTM `0.3623`, XGBoost `0.5506`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `101.5`, LSTM `0.3129`, XGBoost `0.4997`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `99.5`, LSTM `0.3145`, XGBoost `0.4996`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
