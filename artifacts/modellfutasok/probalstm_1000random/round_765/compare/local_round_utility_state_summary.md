# Local Round Utility State Analysis

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-faze-vs-virtuspro-bo3-YDlVsCnS6YPgcr85bBYoPq/faze-vs-virtus-pro-m2-mirage.csv`
- round_num: `5`
- rows: `240`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 240 | 1.000 | 0.027167 | 0.084201 | -0.057033 | 240 | 0 | 1.000000 | 1.000000 |
| active/recent utility | 240 | 1.000 | 0.027167 | 0.084201 | -0.057033 | 240 | 0 | 1.000000 | 1.000000 |
| strong utility action | 130 | 0.542 | 0.033424 | 0.088980 | -0.055556 | 130 | 0 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 130 | 0.542 | 0.033424 | 0.088980 | -0.055556 | 130 | 0 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 240 | 1.000 | 0.027167 | 0.084201 | -0.057033 | 240 | 0 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `5.5s` - `30.5s`, rows `51`
- `51.5s` - `90.5s`, rows `79`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `74.5`, LSTM `0.0349`, XGBoost `0.1876`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `74.0`, LSTM `0.0360`, XGBoost `0.1885`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `75.0`, LSTM `0.0310`, XGBoost `0.1740`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `73.5`, LSTM `0.0507`, XGBoost `0.1876`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `73.0`, LSTM `0.0528`, XGBoost `0.1885`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `72.5`, LSTM `0.0564`, XGBoost `0.1882`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `67.0`, LSTM `0.0423`, XGBoost `0.1685`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `70.5`, LSTM `0.0448`, XGBoost `0.1694`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `66.5`, LSTM `0.0423`, XGBoost `0.1666`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `67.5`, LSTM `0.0493`, XGBoost `0.1717`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
