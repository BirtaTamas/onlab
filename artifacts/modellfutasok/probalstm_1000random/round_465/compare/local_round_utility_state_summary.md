# Local Round Utility State Analysis

- csv_path: `processed_full/blast_open_london/blast-open-london-2025-vitality-vs-faze-bo3-cXlexQJNK-6GX9ddkYcv53/vitality-vs-faze-m1-mirage.csv`
- round_num: `3`
- rows: `179`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 179 | 1.000 | 0.021027 | 0.033662 | -0.012635 | 174 | 5 | 1.000000 | 1.000000 |
| active/recent utility | 179 | 1.000 | 0.021027 | 0.033662 | -0.012635 | 174 | 5 | 1.000000 | 1.000000 |
| strong utility action | 133 | 0.743 | 0.024112 | 0.036392 | -0.012280 | 132 | 1 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 133 | 0.743 | 0.024112 | 0.036392 | -0.012280 | 132 | 1 | 1.000000 | 1.000000 |
| recent utility last 5s | 10 | 0.056 | 0.010928 | 0.033833 | -0.022904 | 10 | 0 | 1.000000 | 1.000000 |
| flash effect present | 179 | 1.000 | 0.021027 | 0.033662 | -0.012635 | 174 | 5 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `8.5s` - `74.5s`, rows `133`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `41.0`, LSTM `0.0053`, XGBoost `0.0413`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `16.0`, recent_utility `0`
- seconds `41.5`, LSTM `0.0056`, XGBoost `0.0410`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `16.0`, recent_utility `0`
- seconds `42.0`, LSTM `0.0059`, XGBoost `0.0410`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `16.0`, recent_utility `0`
- seconds `42.5`, LSTM `0.0059`, XGBoost `0.0410`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `16.0`, recent_utility `0`
- seconds `44.0`, LSTM `0.0062`, XGBoost `0.0411`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `16.0`, recent_utility `0`
- seconds `43.0`, LSTM `0.0066`, XGBoost `0.0410`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `16.0`, recent_utility `0`
- seconds `45.0`, LSTM `0.0074`, XGBoost `0.0414`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `44.5`, LSTM `0.0073`, XGBoost `0.0411`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `8.0`, recent_utility `0`
- seconds `45.5`, LSTM `0.0078`, XGBoost `0.0414`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `46.0`, LSTM `0.0066`, XGBoost `0.0401`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
