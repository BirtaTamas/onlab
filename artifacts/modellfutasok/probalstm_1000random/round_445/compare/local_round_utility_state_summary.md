# Local Round Utility State Analysis

- csv_path: `processed_full/blast_rivals_season_1/blast-rivals-2025-season-1-wildcard-vs-spirit-bo3-VLdaQLy-otUvCLBOl-LFGy/wildcard-vs-spirit-m2-dust2.csv`
- round_num: `10`
- rows: `184`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 184 | 1.000 | 0.196923 | 0.298080 | -0.101157 | 180 | 4 | 1.000000 | 1.000000 |
| active/recent utility | 184 | 1.000 | 0.196923 | 0.298080 | -0.101157 | 180 | 4 | 1.000000 | 1.000000 |
| strong utility action | 155 | 0.842 | 0.191473 | 0.289762 | -0.098289 | 151 | 4 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 149 | 0.810 | 0.191889 | 0.287885 | -0.095996 | 145 | 4 | 1.000000 | 1.000000 |
| recent utility last 5s | 11 | 0.060 | 0.158175 | 0.335385 | -0.177210 | 11 | 0 | 1.000000 | 1.000000 |
| flash effect present | 184 | 1.000 | 0.196923 | 0.298080 | -0.101157 | 180 | 4 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `3.5s` - `34.0s`, rows `62`
- `48.5s` - `91.5s`, rows `87`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `5.0`, LSTM `0.1145`, XGBoost `0.3340`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `2`
- seconds `4.5`, LSTM `0.1186`, XGBoost `0.3340`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `2`
- seconds `5.5`, LSTM `0.1321`, XGBoost `0.3340`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `4.0`, LSTM `0.1338`, XGBoost `0.3340`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `2`
- seconds `3.0`, LSTM `0.1534`, XGBoost `0.3373`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `2`
- seconds `3.5`, LSTM `0.1541`, XGBoost `0.3352`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `2`
- seconds `8.0`, LSTM `0.1585`, XGBoost `0.3337`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `2.5`, LSTM `0.1628`, XGBoost `0.3373`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `2`
- seconds `7.5`, LSTM `0.1609`, XGBoost `0.3340`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `6.0`, LSTM `0.1673`, XGBoost `0.3340`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
