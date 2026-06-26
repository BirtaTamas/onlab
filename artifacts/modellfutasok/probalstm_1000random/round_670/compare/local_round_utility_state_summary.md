# Local Round Utility State Analysis

- csv_path: `processed_full/fissure_playground_1/fissure-playground-1-pain-vs-rare-atom-bo3-Rmb0_mvtIpTmOfUIJVjwOw/pain-vs-rare-atom-m2-dust2.csv`
- round_num: `15`
- rows: `126`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 126 | 1.000 | 0.602818 | 0.572144 | 0.030674 | 101 | 25 | 0.960317 | 0.468254 |
| active/recent utility | 126 | 1.000 | 0.602818 | 0.572144 | 0.030674 | 101 | 25 | 0.960317 | 0.468254 |
| strong utility action | 84 | 0.667 | 0.588148 | 0.542632 | 0.045517 | 80 | 4 | 1.000000 | 0.440476 |
| utility damage | 30 | 0.238 | 0.573958 | 0.513548 | 0.060410 | 30 | 0 | 1.000000 | 0.666667 |
| active smoke/inferno | 84 | 0.667 | 0.588148 | 0.542632 | 0.045517 | 80 | 4 | 1.000000 | 0.440476 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 126 | 1.000 | 0.602818 | 0.572144 | 0.030674 | 101 | 25 | 0.960317 | 0.468254 |

## Active Smoke/Inferno Intervals

- `4.5s` - `39.0s`, rows `70`
- `48.0s` - `54.5s`, rows `14`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `37.0`, LSTM `0.6520`, XGBoost `0.5057`, closer `lstm`, smoke `0`, inferno `2`, utility_damage `12.0`, recent_utility `0`
- seconds `36.0`, LSTM `0.6507`, XGBoost `0.5059`, closer `lstm`, smoke `0`, inferno `2`, utility_damage `12.0`, recent_utility `0`
- seconds `37.5`, LSTM `0.6489`, XGBoost `0.5055`, closer `lstm`, smoke `0`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `36.5`, LSTM `0.6431`, XGBoost `0.5059`, closer `lstm`, smoke `0`, inferno `2`, utility_damage `12.0`, recent_utility `0`
- seconds `38.0`, LSTM `0.6343`, XGBoost `0.5055`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `35.5`, LSTM `0.6260`, XGBoost `0.5092`, closer `lstm`, smoke `1`, inferno `2`, utility_damage `12.0`, recent_utility `0`
- seconds `48.0`, LSTM `0.5855`, XGBoost `0.4742`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `53.0`, LSTM `0.8008`, XGBoost `0.7086`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `53.5`, LSTM `0.7909`, XGBoost `0.7083`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `49.5`, LSTM `0.6340`, XGBoost `0.7142`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
