# Local Round Utility State Analysis

- csv_path: `processed_full/blast_open_lisbon/blast-open-lisbon-2025-eternal-fire-vs-natus-vincere-bo3-TFptrqwLQ_nOvi5zixIc9R/eternal-fire-vs-natus-vincere-m2-dust2.csv`
- round_num: `18`
- rows: `107`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 107 | 1.000 | 0.175624 | 0.253220 | -0.077595 | 107 | 0 | 1.000000 | 1.000000 |
| active/recent utility | 107 | 1.000 | 0.175624 | 0.253220 | -0.077595 | 107 | 0 | 1.000000 | 1.000000 |
| strong utility action | 97 | 0.907 | 0.167618 | 0.238743 | -0.071125 | 97 | 0 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 87 | 0.813 | 0.152367 | 0.219878 | -0.067511 | 87 | 0 | 1.000000 | 1.000000 |
| recent utility last 5s | 10 | 0.093 | 0.300303 | 0.402871 | -0.102568 | 10 | 0 | 1.000000 | 1.000000 |
| flash effect present | 107 | 1.000 | 0.175624 | 0.253220 | -0.077595 | 107 | 0 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `10.0s` - `53.0s`, rows `87`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `10.0`, LSTM `0.2508`, XGBoost `0.3983`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `31.0`, LSTM `0.1103`, XGBoost `0.2423`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `11.0`, LSTM `0.2719`, XGBoost `0.3983`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `5.5`, LSTM `0.2755`, XGBoost `0.4016`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `24.0`, LSTM `0.2860`, XGBoost `0.4109`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `21.5`, LSTM `0.2774`, XGBoost `0.4014`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `12.5`, LSTM `0.2765`, XGBoost `0.3983`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `2.0`, LSTM `0.2817`, XGBoost `0.4028`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `2.5`, LSTM `0.2818`, XGBoost `0.4028`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `21.0`, LSTM `0.2812`, XGBoost `0.4014`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
