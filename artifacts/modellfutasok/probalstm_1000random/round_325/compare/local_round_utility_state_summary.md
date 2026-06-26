# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major_stage_2/blasttv-austin-major-2025-stage-2-b8-vs-lynn-vision-bo3-Whl3pjYuIoHffY1VOn8vws/b8-vs-lynn-vision-m1-dust2.csv`
- round_num: `10`
- rows: `211`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 211 | 1.000 | 0.194216 | 0.306033 | -0.111818 | 201 | 10 | 0.990521 | 0.990521 |
| active/recent utility | 211 | 1.000 | 0.194216 | 0.306033 | -0.111818 | 201 | 10 | 0.990521 | 0.990521 |
| strong utility action | 115 | 0.545 | 0.190347 | 0.294706 | -0.104359 | 115 | 0 | 1.000000 | 0.991304 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 115 | 0.545 | 0.190347 | 0.294706 | -0.104359 | 115 | 0 | 1.000000 | 0.991304 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 211 | 1.000 | 0.194216 | 0.306033 | -0.111818 | 201 | 10 | 0.990521 | 0.990521 |

## Active Smoke/Inferno Intervals

- `10.5s` - `48.5s`, rows `77`
- `86.5s` - `105.0s`, rows `38`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `10.5`, LSTM `0.1465`, XGBoost `0.3462`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `11.0`, LSTM `0.1568`, XGBoost `0.3462`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `11.5`, LSTM `0.1656`, XGBoost `0.3462`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `16.0`, LSTM `0.1800`, XGBoost `0.3471`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `12.0`, LSTM `0.1869`, XGBoost `0.3462`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `12.5`, LSTM `0.1898`, XGBoost `0.3483`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `15.5`, LSTM `0.1928`, XGBoost `0.3499`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `15.0`, LSTM `0.1951`, XGBoost `0.3499`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `100.0`, LSTM `0.1207`, XGBoost `0.2734`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `14.0`, LSTM `0.1981`, XGBoost `0.3492`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
