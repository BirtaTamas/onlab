# Local Round Utility State Analysis

- csv_path: `processed_full/blast_open_lisbon/blast-open-lisbon-2025-faze-vs-virtuspro-bo3-YK5CkfojMMEdQ3U0_CVA2l/faze-vs-virtuspro-m1-anubis.csv`
- round_num: `16`
- rows: `191`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 191 | 1.000 | 0.078927 | 0.155191 | -0.076264 | 191 | 0 | 1.000000 | 1.000000 |
| active/recent utility | 191 | 1.000 | 0.078927 | 0.155191 | -0.076264 | 191 | 0 | 1.000000 | 1.000000 |
| strong utility action | 156 | 0.817 | 0.091559 | 0.175238 | -0.083678 | 156 | 0 | 1.000000 | 1.000000 |
| utility damage | 10 | 0.052 | 0.166053 | 0.267491 | -0.101437 | 10 | 0 | 1.000000 | 1.000000 |
| active smoke/inferno | 144 | 0.754 | 0.086794 | 0.162801 | -0.076008 | 144 | 0 | 1.000000 | 1.000000 |
| recent utility last 5s | 12 | 0.063 | 0.148749 | 0.324475 | -0.175726 | 12 | 0 | 1.000000 | 1.000000 |
| flash effect present | 191 | 1.000 | 0.078927 | 0.155191 | -0.076264 | 191 | 0 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `8.5s` - `80.0s`, rows `144`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `1.0`, LSTM `0.0823`, XGBoost `0.3325`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `0.5`, LSTM `0.0879`, XGBoost `0.3325`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `1.5`, LSTM `0.1072`, XGBoost `0.3344`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `2`
- seconds `2.0`, LSTM `0.1230`, XGBoost `0.3287`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `2`
- seconds `2.5`, LSTM `0.1365`, XGBoost `0.3227`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `2`
- seconds `3.0`, LSTM `0.1436`, XGBoost `0.3216`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `2`
- seconds `46.5`, LSTM `0.0460`, XGBoost `0.2178`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `47.0`, LSTM `0.0544`, XGBoost `0.2169`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `48.0`, LSTM `0.0493`, XGBoost `0.2105`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `3.5`, LSTM `0.1599`, XGBoost `0.3207`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `2`
