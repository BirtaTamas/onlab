# Local Round Utility State Analysis

- csv_path: `processed_full/blast_rivals_season_1/blast-rivals-2025-season-1-spirit-vs-flyquest-bo3-fQI-qOiPd1cRkmhkz0Xs5h/spirit-vs-flyquest-m1-mirage.csv`
- round_num: `8`
- rows: `150`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 150 | 1.000 | 0.220581 | 0.298172 | -0.077591 | 8 | 142 | 0.140000 | 0.306667 |
| active/recent utility | 150 | 1.000 | 0.220581 | 0.298172 | -0.077591 | 8 | 142 | 0.140000 | 0.306667 |
| strong utility action | 140 | 0.933 | 0.217482 | 0.288890 | -0.071408 | 7 | 133 | 0.150000 | 0.285714 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 126 | 0.840 | 0.226437 | 0.299629 | -0.073192 | 7 | 119 | 0.166667 | 0.317460 |
| recent utility last 5s | 16 | 0.107 | 0.136891 | 0.192894 | -0.056003 | 0 | 16 | 0.000000 | 0.000000 |
| flash effect present | 150 | 1.000 | 0.220581 | 0.298172 | -0.077591 | 8 | 142 | 0.140000 | 0.306667 |

## Active Smoke/Inferno Intervals

- `7.5s` - `49.0s`, rows `84`
- `54.0s` - `74.5s`, rows `42`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `69.0`, LSTM `0.5445`, XGBoost `0.8346`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `69.5`, LSTM `0.6049`, XGBoost `0.8346`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `68.5`, LSTM `0.6085`, XGBoost `0.8362`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `67.0`, LSTM `0.4648`, XGBoost `0.6854`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `66.0`, LSTM `0.4759`, XGBoost `0.6854`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `57.5`, LSTM `0.3819`, XGBoost `0.5863`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `55.5`, LSTM `0.3744`, XGBoost `0.5762`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `64.5`, LSTM `0.4831`, XGBoost `0.6842`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `74.0`, LSTM `0.6331`, XGBoost `0.8342`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `73.5`, LSTM `0.6330`, XGBoost `0.8338`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
