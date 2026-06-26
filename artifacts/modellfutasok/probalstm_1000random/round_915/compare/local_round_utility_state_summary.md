# Local Round Utility State Analysis

- csv_path: `processed_full/blast_open_lisbon/blast-open-lisbon-2025-spirit-vs-the-huns-bo3-TWIJIxJZifB3vPv3OUvjVr/spirit-vs-the-huns-m2-dust2.csv`
- round_num: `17`
- rows: `207`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 207 | 1.000 | 0.399465 | 0.517382 | -0.117917 | 0 | 207 | 0.149758 | 0.159420 |
| active/recent utility | 207 | 1.000 | 0.399465 | 0.517382 | -0.117917 | 0 | 207 | 0.149758 | 0.159420 |
| strong utility action | 170 | 0.821 | 0.413861 | 0.531978 | -0.118117 | 0 | 170 | 0.182353 | 0.194118 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 170 | 0.821 | 0.413861 | 0.531978 | -0.118117 | 0 | 170 | 0.182353 | 0.194118 |
| recent utility last 5s | 10 | 0.048 | 0.288399 | 0.444088 | -0.155690 | 0 | 10 | 0.000000 | 0.000000 |
| flash effect present | 207 | 1.000 | 0.399465 | 0.517382 | -0.117917 | 0 | 207 | 0.149758 | 0.159420 |

## Active Smoke/Inferno Intervals

- `8.0s` - `35.0s`, rows `55`
- `45.0s` - `52.5s`, rows `16`
- `54.0s` - `103.0s`, rows `99`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `23.5`, LSTM `0.2567`, XGBoost `0.4721`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `24.0`, LSTM `0.2696`, XGBoost `0.4721`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `22.0`, LSTM `0.2720`, XGBoost `0.4733`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `22.5`, LSTM `0.2784`, XGBoost `0.4735`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `82.5`, LSTM `0.2527`, XGBoost `0.4416`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `23.0`, LSTM `0.2838`, XGBoost `0.4721`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `24.5`, LSTM `0.2852`, XGBoost `0.4721`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `21.5`, LSTM `0.2769`, XGBoost `0.4638`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `86.5`, LSTM `0.2401`, XGBoost `0.4199`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `33.0`, LSTM `0.2717`, XGBoost `0.4462`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
