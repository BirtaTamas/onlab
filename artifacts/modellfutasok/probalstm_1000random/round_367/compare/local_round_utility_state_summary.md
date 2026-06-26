# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major_stage_2/blasttv-austin-major-2025-stage-2-faze-vs-heroic-dust2-PtQF8ASKD1754yZQHk6148/faze-vs-heroic-dust2.csv`
- round_num: `2`
- rows: `175`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 175 | 1.000 | 0.938412 | 0.986575 | -0.048163 | 0 | 175 | 1.000000 | 1.000000 |
| active/recent utility | 175 | 1.000 | 0.938412 | 0.986575 | -0.048163 | 0 | 175 | 1.000000 | 1.000000 |
| strong utility action | 50 | 0.286 | 0.900876 | 0.979034 | -0.078158 | 0 | 50 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 50 | 0.286 | 0.900876 | 0.979034 | -0.078158 | 0 | 50 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 175 | 1.000 | 0.938412 | 0.986575 | -0.048163 | 0 | 175 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `8.5s` - `33.0s`, rows `50`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `32.0`, LSTM `0.8825`, XGBoost `0.9798`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `31.0`, LSTM `0.8823`, XGBoost `0.9795`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `29.5`, LSTM `0.8828`, XGBoost `0.9793`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `8.5`, LSTM `0.8825`, XGBoost `0.9786`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `29.0`, LSTM `0.8846`, XGBoost `0.9793`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `30.0`, LSTM `0.8859`, XGBoost `0.9793`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `33.0`, LSTM `0.8865`, XGBoost `0.9798`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `24.0`, LSTM `0.8893`, XGBoost `0.9792`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `31.5`, LSTM `0.8900`, XGBoost `0.9798`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `21.5`, LSTM `0.8894`, XGBoost `0.9792`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
