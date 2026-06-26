# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-legacy-vs-lynn-vision-bo3-80tf5tBYONxHYQuFp0AoSQ/legacy-vs-lynn-vision-m1-dust2.csv`
- round_num: `6`
- rows: `216`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 216 | 1.000 | 0.329437 | 0.344264 | -0.014827 | 158 | 58 | 0.805556 | 0.958333 |
| active/recent utility | 216 | 1.000 | 0.329437 | 0.344264 | -0.014827 | 158 | 58 | 0.805556 | 0.958333 |
| strong utility action | 174 | 0.806 | 0.343352 | 0.360687 | -0.017335 | 130 | 44 | 0.810345 | 0.948276 |
| utility damage | 10 | 0.046 | 0.575970 | 0.487090 | 0.088880 | 0 | 10 | 0.000000 | 1.000000 |
| active smoke/inferno | 174 | 0.806 | 0.343352 | 0.360687 | -0.017335 | 130 | 44 | 0.810345 | 0.948276 |
| recent utility last 5s | 20 | 0.093 | 0.374322 | 0.409734 | -0.035412 | 15 | 5 | 0.950000 | 1.000000 |
| flash effect present | 216 | 1.000 | 0.329437 | 0.344264 | -0.014827 | 158 | 58 | 0.805556 | 0.958333 |

## Active Smoke/Inferno Intervals

- `11.0s` - `97.5s`, rows `174`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `65.0`, LSTM `0.4909`, XGBoost `0.3738`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `46.0`, LSTM `0.6060`, XGBoost `0.4900`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `3.0`, recent_utility `0`
- seconds `46.5`, LSTM `0.6053`, XGBoost `0.4900`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `3.0`, recent_utility `0`
- seconds `45.0`, LSTM `0.5936`, XGBoost `0.4875`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `3.0`, recent_utility `0`
- seconds `44.5`, LSTM `0.5921`, XGBoost `0.4875`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `3.0`, recent_utility `0`
- seconds `63.0`, LSTM `0.1335`, XGBoost `0.2345`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `45.5`, LSTM `0.5837`, XGBoost `0.4875`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `3.0`, recent_utility `0`
- seconds `70.5`, LSTM `0.5013`, XGBoost `0.4071`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `25.5`, LSTM `0.4026`, XGBoost `0.4935`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `47.0`, LSTM `0.5805`, XGBoost `0.4900`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `3.0`, recent_utility `0`
