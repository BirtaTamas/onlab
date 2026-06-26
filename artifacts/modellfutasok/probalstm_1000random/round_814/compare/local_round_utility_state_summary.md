# Local Round Utility State Analysis

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-heroic-vs-natus-vincere-bo3-P_vZ7pAIyzYcLTUjDHhSUR/heroic-vs-natus-vincere-m2-ancient.csv`
- round_num: `2`
- rows: `169`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 169 | 1.000 | 0.031248 | 0.083324 | -0.052077 | 168 | 1 | 1.000000 | 1.000000 |
| active/recent utility | 169 | 1.000 | 0.031248 | 0.083324 | -0.052077 | 168 | 1 | 1.000000 | 1.000000 |
| strong utility action | 122 | 0.722 | 0.019496 | 0.073792 | -0.054296 | 122 | 0 | 1.000000 | 1.000000 |
| utility damage | 10 | 0.059 | 0.052656 | 0.154197 | -0.101541 | 10 | 0 | 1.000000 | 1.000000 |
| active smoke/inferno | 122 | 0.722 | 0.019496 | 0.073792 | -0.054296 | 122 | 0 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 169 | 1.000 | 0.031248 | 0.083324 | -0.052077 | 168 | 1 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `6.5s` - `29.5s`, rows `47`
- `35.0s` - `72.0s`, rows `75`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `24.5`, LSTM `0.0198`, XGBoost `0.2313`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `6.5`, LSTM `0.1326`, XGBoost `0.3012`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `7.0`, LSTM `0.1664`, XGBoost `0.2987`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `19.5`, LSTM `0.0256`, XGBoost `0.1531`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `19.0`, LSTM `0.0286`, XGBoost `0.1531`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `13.0`, LSTM `0.0268`, XGBoost `0.1456`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `22.0`, recent_utility `0`
- seconds `12.5`, LSTM `0.0265`, XGBoost `0.1454`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `22.0`, recent_utility `0`
- seconds `18.5`, LSTM `0.0327`, XGBoost `0.1505`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `13.5`, LSTM `0.0299`, XGBoost `0.1465`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `22.0`, recent_utility `0`
- seconds `12.0`, LSTM `0.0279`, XGBoost `0.1429`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `22.0`, recent_utility `0`
