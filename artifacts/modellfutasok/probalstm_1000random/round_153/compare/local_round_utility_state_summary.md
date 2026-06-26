# Local Round Utility State Analysis

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-vitality-vs-virtuspro-bo3-8Z0L17IYJlstHvIADVy9G9/vitality-vs-virtus-pro-m3-mirage.csv`
- round_num: `13`
- rows: `137`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 137 | 1.000 | 0.455094 | 0.495624 | -0.040531 | 115 | 22 | 0.364964 | 0.291971 |
| active/recent utility | 137 | 1.000 | 0.455094 | 0.495624 | -0.040531 | 115 | 22 | 0.364964 | 0.291971 |
| strong utility action | 84 | 0.613 | 0.424145 | 0.480523 | -0.056377 | 80 | 4 | 0.392857 | 0.285714 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 84 | 0.613 | 0.424145 | 0.480523 | -0.056377 | 80 | 4 | 0.392857 | 0.285714 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 137 | 1.000 | 0.455094 | 0.495624 | -0.040531 | 115 | 22 | 0.364964 | 0.291971 |

## Active Smoke/Inferno Intervals

- `19.0s` - `40.5s`, rows `44`
- `48.5s` - `68.0s`, rows `40`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `60.0`, LSTM `0.5402`, XGBoost `0.7421`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `60.5`, LSTM `0.3681`, XGBoost `0.5601`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `68.0`, LSTM `0.0326`, XGBoost `0.2193`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `57.0`, LSTM `0.5714`, XGBoost `0.7508`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `57.5`, LSTM `0.5731`, XGBoost `0.7508`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `56.5`, LSTM `0.6656`, XGBoost `0.8365`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `59.5`, LSTM `0.7365`, XGBoost `0.8925`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `65.0`, LSTM `0.0287`, XGBoost `0.1830`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `59.0`, LSTM `0.7345`, XGBoost `0.8880`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `64.5`, LSTM `0.0286`, XGBoost `0.1814`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
