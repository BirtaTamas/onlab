# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-legacy-vs-lynn-vision-bo3-80tf5tBYONxHYQuFp0AoSQ/legacy-vs-lynn-vision-m1-dust2.csv`
- round_num: `8`
- rows: `273`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 273 | 1.000 | 0.099062 | 0.152356 | -0.053293 | 273 | 0 | 1.000000 | 1.000000 |
| active/recent utility | 273 | 1.000 | 0.099062 | 0.152356 | -0.053293 | 273 | 0 | 1.000000 | 1.000000 |
| strong utility action | 241 | 0.883 | 0.088388 | 0.142388 | -0.054000 | 241 | 0 | 1.000000 | 1.000000 |
| utility damage | 10 | 0.037 | 0.047459 | 0.159190 | -0.111732 | 10 | 0 | 1.000000 | 1.000000 |
| active smoke/inferno | 231 | 0.846 | 0.087451 | 0.138843 | -0.051392 | 231 | 0 | 1.000000 | 1.000000 |
| recent utility last 5s | 10 | 0.037 | 0.110046 | 0.224281 | -0.114235 | 10 | 0 | 1.000000 | 1.000000 |
| flash effect present | 273 | 1.000 | 0.099062 | 0.152356 | -0.053293 | 273 | 0 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `10.5s` - `34.0s`, rows `48`
- `45.0s` - `136.0s`, rows `183`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `82.0`, LSTM `0.1015`, XGBoost `0.2644`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `30.0`, recent_utility `0`
- seconds `82.5`, LSTM `0.1101`, XGBoost `0.2644`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `30.0`, recent_utility `0`
- seconds `81.0`, LSTM `0.0429`, XGBoost `0.1785`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `81.5`, LSTM `0.0436`, XGBoost `0.1775`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `1.5`, LSTM `0.0905`, XGBoost `0.2231`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `2.5`, LSTM `0.0944`, XGBoost `0.2215`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `2.0`, LSTM `0.0945`, XGBoost `0.2215`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `80.5`, LSTM `0.0530`, XGBoost `0.1785`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `1.0`, LSTM `0.0982`, XGBoost `0.2231`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `3.0`, LSTM `0.1009`, XGBoost `0.2220`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
