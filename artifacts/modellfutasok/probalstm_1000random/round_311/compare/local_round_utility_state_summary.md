# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-legacy-vs-lynn-vision-bo3-80tf5tBYONxHYQuFp0AoSQ/legacy-vs-lynn-vision-m2-inferno.csv`
- round_num: `6`
- rows: `170`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 170 | 1.000 | 0.903312 | 0.885938 | 0.017373 | 131 | 39 | 1.000000 | 1.000000 |
| active/recent utility | 170 | 1.000 | 0.903312 | 0.885938 | 0.017373 | 131 | 39 | 1.000000 | 1.000000 |
| strong utility action | 122 | 0.718 | 0.942438 | 0.931317 | 0.011120 | 83 | 39 | 1.000000 | 1.000000 |
| utility damage | 21 | 0.124 | 0.882614 | 0.849111 | 0.033503 | 20 | 1 | 1.000000 | 1.000000 |
| active smoke/inferno | 122 | 0.718 | 0.942438 | 0.931317 | 0.011120 | 83 | 39 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 170 | 1.000 | 0.903312 | 0.885938 | 0.017373 | 131 | 39 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `10.0s` - `42.5s`, rows `66`
- `57.0s` - `84.5s`, rows `56`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `19.0`, LSTM `0.8954`, XGBoost `0.7695`, closer `lstm`, smoke `3`, inferno `2`, utility_damage `8.0`, recent_utility `0`
- seconds `11.5`, LSTM `0.6372`, XGBoost `0.5202`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `2.0`, recent_utility `0`
- seconds `11.0`, LSTM `0.6339`, XGBoost `0.5281`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `2.0`, recent_utility `0`
- seconds `10.5`, LSTM `0.6288`, XGBoost `0.5339`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `1.0`, recent_utility `0`
- seconds `12.0`, LSTM `0.6365`, XGBoost `0.5429`, closer `lstm`, smoke `2`, inferno `3`, utility_damage `2.0`, recent_utility `0`
- seconds `10.0`, LSTM `0.6038`, XGBoost `0.5452`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `40.5`, LSTM `0.9726`, XGBoost `0.9407`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `41.0`, LSTM `0.9712`, XGBoost `0.9402`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `15.0`, LSTM `0.9403`, XGBoost `0.9098`, closer `lstm`, smoke `2`, inferno `3`, utility_damage `17.0`, recent_utility `0`
- seconds `39.5`, LSTM `0.9713`, XGBoost `0.9407`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
