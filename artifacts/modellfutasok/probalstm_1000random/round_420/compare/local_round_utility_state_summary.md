# Local Round Utility State Analysis

- csv_path: `processed_full/blast_open_lisbon/blast-open-lisbon-2025-imperial-vs-liquid-bo3-eiIGPV5tjvJFQ73hC8D8JI/imperial-vs-liquid-m3-anubis.csv`
- round_num: `8`
- rows: `190`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 190 | 1.000 | 0.405933 | 0.427314 | -0.021382 | 120 | 70 | 0.426316 | 0.352632 |
| active/recent utility | 190 | 1.000 | 0.405933 | 0.427314 | -0.021382 | 120 | 70 | 0.426316 | 0.352632 |
| strong utility action | 142 | 0.747 | 0.492806 | 0.504034 | -0.011228 | 72 | 70 | 0.246479 | 0.232394 |
| utility damage | 10 | 0.053 | 0.551632 | 0.515589 | 0.036044 | 0 | 10 | 0.000000 | 0.000000 |
| active smoke/inferno | 142 | 0.747 | 0.492806 | 0.504034 | -0.011228 | 72 | 70 | 0.246479 | 0.232394 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 190 | 1.000 | 0.405933 | 0.427314 | -0.021382 | 120 | 70 | 0.426316 | 0.352632 |

## Active Smoke/Inferno Intervals

- `7.0s` - `77.5s`, rows `142`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `65.5`, LSTM `0.4428`, XGBoost `0.2889`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `45.0`, LSTM `0.6122`, XGBoost `0.7478`, closer `lstm`, smoke `6`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `45.5`, LSTM `0.6181`, XGBoost `0.7474`, closer `lstm`, smoke `6`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `46.0`, LSTM `0.6249`, XGBoost `0.7474`, closer `lstm`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `46.5`, LSTM `0.6399`, XGBoost `0.7478`, closer `lstm`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `47.0`, LSTM `0.6550`, XGBoost `0.7478`, closer `lstm`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `55.0`, LSTM `0.6842`, XGBoost `0.7679`, closer `lstm`, smoke `5`, inferno `2`, utility_damage `36.0`, recent_utility `0`
- seconds `69.0`, LSTM `0.2114`, XGBoost `0.2927`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `60.0`, LSTM `0.5583`, XGBoost `0.6327`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `60.5`, LSTM `0.5609`, XGBoost `0.6328`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
