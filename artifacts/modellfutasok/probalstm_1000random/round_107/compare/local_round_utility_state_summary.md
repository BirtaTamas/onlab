# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major_stage_2/blasttv-austin-major-2025-stage-2-furia-vs-b8-bo3-3h93b_qbGndTgDFTW66Ud1/furia-vs-b8-m1-mirage.csv`
- round_num: `6`
- rows: `196`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 196 | 1.000 | 0.474560 | 0.553110 | -0.078550 | 135 | 61 | 0.331633 | 0.311224 |
| active/recent utility | 196 | 1.000 | 0.474560 | 0.553110 | -0.078550 | 135 | 61 | 0.331633 | 0.311224 |
| strong utility action | 151 | 0.770 | 0.533080 | 0.581026 | -0.047946 | 102 | 49 | 0.245033 | 0.218543 |
| utility damage | 10 | 0.051 | 0.571026 | 0.616438 | -0.045412 | 10 | 0 | 0.000000 | 0.000000 |
| active smoke/inferno | 151 | 0.770 | 0.533080 | 0.581026 | -0.047946 | 102 | 49 | 0.245033 | 0.218543 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 196 | 1.000 | 0.474560 | 0.553110 | -0.078550 | 135 | 61 | 0.331633 | 0.311224 |

## Active Smoke/Inferno Intervals

- `6.0s` - `48.5s`, rows `86`
- `51.5s` - `83.5s`, rows `65`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `67.0`, LSTM `0.1870`, XGBoost `0.4571`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `67.5`, LSTM `0.1903`, XGBoost `0.4569`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `78.0`, LSTM `0.1848`, XGBoost `0.4317`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `83.5`, LSTM `0.4175`, XGBoost `0.6456`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `69.0`, LSTM `0.2391`, XGBoost `0.4616`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `69.5`, LSTM `0.2397`, XGBoost `0.4616`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `68.5`, LSTM `0.2372`, XGBoost `0.4562`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `79.0`, LSTM `0.2048`, XGBoost `0.4230`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `66.5`, LSTM `0.2387`, XGBoost `0.4569`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `77.0`, LSTM `0.2448`, XGBoost `0.4540`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
