# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major_stage_2/blasttv-austin-major-2025-stage-2-furia-vs-b8-bo3-3h93b_qbGndTgDFTW66Ud1/furia-vs-b8-m1-mirage.csv`
- round_num: `11`
- rows: `198`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 198 | 1.000 | 0.361632 | 0.449504 | -0.087871 | 17 | 181 | 0.232323 | 0.444444 |
| active/recent utility | 198 | 1.000 | 0.361632 | 0.449504 | -0.087871 | 17 | 181 | 0.232323 | 0.444444 |
| strong utility action | 181 | 0.914 | 0.347077 | 0.435049 | -0.087972 | 17 | 164 | 0.232044 | 0.392265 |
| utility damage | 12 | 0.061 | 0.512367 | 0.529194 | -0.016827 | 0 | 12 | 0.833333 | 1.000000 |
| active smoke/inferno | 181 | 0.914 | 0.347077 | 0.435049 | -0.087972 | 17 | 164 | 0.232044 | 0.392265 |
| recent utility last 5s | 10 | 0.051 | 0.311209 | 0.343856 | -0.032647 | 1 | 9 | 0.000000 | 0.000000 |
| flash effect present | 198 | 1.000 | 0.361632 | 0.449504 | -0.087871 | 17 | 181 | 0.232323 | 0.444444 |

## Active Smoke/Inferno Intervals

- `6.5s` - `94.0s`, rows `176`
- `96.5s` - `98.5s`, rows `5`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `80.5`, LSTM `0.1508`, XGBoost `0.4423`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `80.0`, LSTM `0.1547`, XGBoost `0.4423`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `81.0`, LSTM `0.1542`, XGBoost `0.4412`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `82.0`, LSTM `0.1563`, XGBoost `0.4412`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `81.5`, LSTM `0.1566`, XGBoost `0.4412`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `79.0`, LSTM `0.1652`, XGBoost `0.4490`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `82.5`, LSTM `0.1586`, XGBoost `0.4412`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `84.5`, LSTM `0.5617`, XGBoost `0.8435`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `83.0`, LSTM `0.1605`, XGBoost `0.4412`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `83.5`, LSTM `0.1618`, XGBoost `0.4412`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
