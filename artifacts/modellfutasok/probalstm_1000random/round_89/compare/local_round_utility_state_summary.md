# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-legacy-vs-lynn-vision-bo3-80tf5tBYONxHYQuFp0AoSQ/legacy-vs-lynn-vision-m1-dust2.csv`
- round_num: `10`
- rows: `135`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 135 | 1.000 | 0.462814 | 0.444804 | 0.018010 | 30 | 105 | 0.244444 | 0.659259 |
| active/recent utility | 135 | 1.000 | 0.462814 | 0.444804 | 0.018010 | 30 | 105 | 0.244444 | 0.659259 |
| strong utility action | 124 | 0.919 | 0.486368 | 0.464674 | 0.021694 | 21 | 103 | 0.193548 | 0.637097 |
| utility damage | 31 | 0.230 | 0.510746 | 0.475955 | 0.034791 | 3 | 28 | 0.129032 | 0.129032 |
| active smoke/inferno | 119 | 0.881 | 0.483393 | 0.463556 | 0.019837 | 21 | 98 | 0.201681 | 0.621849 |
| recent utility last 5s | 11 | 0.081 | 0.542707 | 0.490359 | 0.052348 | 0 | 11 | 0.000000 | 1.000000 |
| flash effect present | 135 | 1.000 | 0.462814 | 0.444804 | 0.018010 | 30 | 105 | 0.244444 | 0.659259 |

## Active Smoke/Inferno Intervals

- `3.5s` - `62.5s`, rows `119`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `55.5`, LSTM `0.0456`, XGBoost `0.1751`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `61.5`, LSTM `0.4524`, XGBoost `0.5771`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `58.5`, LSTM `0.4390`, XGBoost `0.5628`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `61.0`, LSTM `0.4427`, XGBoost `0.5650`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `62.0`, LSTM `0.4636`, XGBoost `0.5801`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `58.0`, LSTM `0.4528`, XGBoost `0.5638`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `59.5`, LSTM `0.4557`, XGBoost `0.5647`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `56.0`, LSTM `0.0684`, XGBoost `0.1738`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `60.0`, LSTM `0.4586`, XGBoost `0.5552`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `1.0`, LSTM `0.5929`, XGBoost `0.4966`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
