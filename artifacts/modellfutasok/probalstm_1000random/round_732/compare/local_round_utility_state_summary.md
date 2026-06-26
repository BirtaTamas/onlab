# Local Round Utility State Analysis

- csv_path: `processed_full/blast_open_london/blast-open-london-2025-vitality-vs-faze-bo3-cXlexQJNK-6GX9ddkYcv53/vitality-vs-faze-m1-mirage.csv`
- round_num: `26`
- rows: `222`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 222 | 1.000 | 0.462938 | 0.472191 | -0.009253 | 75 | 147 | 0.283784 | 0.238739 |
| active/recent utility | 222 | 1.000 | 0.462938 | 0.472191 | -0.009253 | 75 | 147 | 0.283784 | 0.238739 |
| strong utility action | 207 | 0.932 | 0.469361 | 0.478065 | -0.008705 | 67 | 140 | 0.270531 | 0.231884 |
| utility damage | 21 | 0.095 | 0.033668 | 0.087391 | -0.053722 | 21 | 0 | 1.000000 | 1.000000 |
| active smoke/inferno | 198 | 0.892 | 0.463183 | 0.476049 | -0.012866 | 67 | 131 | 0.282828 | 0.242424 |
| recent utility last 5s | 20 | 0.090 | 0.332417 | 0.353812 | -0.021395 | 10 | 10 | 0.500000 | 0.500000 |
| flash effect present | 222 | 1.000 | 0.462938 | 0.472191 | -0.009253 | 75 | 147 | 0.283784 | 0.238739 |

## Active Smoke/Inferno Intervals

- `6.0s` - `38.0s`, rows `65`
- `39.5s` - `77.5s`, rows `77`
- `80.5s` - `108.0s`, rows `56`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `83.0`, LSTM `0.2385`, XGBoost `0.5613`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `82.0`, LSTM `0.2525`, XGBoost `0.5621`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `82.5`, LSTM `0.2629`, XGBoost `0.5621`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `83.5`, LSTM `0.2219`, XGBoost `0.5199`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `84.0`, LSTM `0.2314`, XGBoost `0.5227`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `84.5`, LSTM `0.2140`, XGBoost `0.4934`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `81.5`, LSTM `0.2900`, XGBoost `0.5626`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `81.0`, LSTM `0.3271`, XGBoost `0.5552`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `80.5`, LSTM `0.3457`, XGBoost `0.5552`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `87.0`, LSTM `0.0435`, XGBoost `0.2028`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
