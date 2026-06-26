# Local Round Utility State Analysis

- csv_path: `processed_full/blast_open_london/blast-open-london-2025-vitality-vs-faze-bo3-cXlexQJNK-6GX9ddkYcv53/vitality-vs-faze-m1-mirage.csv`
- round_num: `22`
- rows: `181`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 181 | 1.000 | 0.647402 | 0.715799 | -0.068397 | 1 | 180 | 0.790055 | 1.000000 |
| active/recent utility | 181 | 1.000 | 0.647402 | 0.715799 | -0.068397 | 1 | 180 | 0.790055 | 1.000000 |
| strong utility action | 119 | 0.657 | 0.583036 | 0.654071 | -0.071035 | 1 | 118 | 0.798319 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 114 | 0.630 | 0.565583 | 0.639061 | -0.073478 | 1 | 113 | 0.789474 | 1.000000 |
| recent utility last 5s | 5 | 0.028 | 0.980956 | 0.996292 | -0.015335 | 0 | 5 | 1.000000 | 1.000000 |
| flash effect present | 181 | 1.000 | 0.647402 | 0.715799 | -0.068397 | 1 | 180 | 0.790055 | 1.000000 |

## Active Smoke/Inferno Intervals

- `7.0s` - `63.5s`, rows `114`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `38.5`, LSTM `0.5471`, XGBoost `0.7785`, closer `xgboost`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `38.0`, LSTM `0.5503`, XGBoost `0.7785`, closer `xgboost`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `39.0`, LSTM `0.5509`, XGBoost `0.7785`, closer `xgboost`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `41.5`, LSTM `0.3529`, XGBoost `0.5602`, closer `xgboost`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `41.0`, LSTM `0.3574`, XGBoost `0.5602`, closer `xgboost`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `40.5`, LSTM `0.3605`, XGBoost `0.5602`, closer `xgboost`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `37.0`, LSTM `0.5835`, XGBoost `0.7785`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `37.5`, LSTM `0.5845`, XGBoost `0.7785`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `31.5`, LSTM `0.5918`, XGBoost `0.7748`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `42.0`, LSTM `0.3780`, XGBoost `0.5602`, closer `xgboost`, smoke `3`, inferno `2`, utility_damage `0.0`, recent_utility `0`
