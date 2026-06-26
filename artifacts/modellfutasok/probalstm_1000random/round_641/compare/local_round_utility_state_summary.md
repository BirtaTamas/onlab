# Local Round Utility State Analysis

- csv_path: `processed_full/iem_cologne/iem-cologne-2025-ninjas-in-pyjamas-vs-natus-vincere-bo3-NyGFAYn592Hu0YfljTVG0N/ninjas-in-pyjamas-vs-natus-vincere-m3-train.csv`
- round_num: `6`
- rows: `216`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 216 | 1.000 | 0.754506 | 0.772372 | -0.017866 | 55 | 161 | 0.995370 | 0.972222 |
| active/recent utility | 216 | 1.000 | 0.754506 | 0.772372 | -0.017866 | 55 | 161 | 0.995370 | 0.972222 |
| strong utility action | 125 | 0.579 | 0.696710 | 0.718507 | -0.021797 | 35 | 90 | 0.992000 | 0.952000 |
| utility damage | 10 | 0.046 | 0.642435 | 0.634023 | 0.008412 | 7 | 3 | 1.000000 | 1.000000 |
| active smoke/inferno | 125 | 0.579 | 0.696710 | 0.718507 | -0.021797 | 35 | 90 | 0.992000 | 0.952000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 216 | 1.000 | 0.754506 | 0.772372 | -0.017866 | 55 | 161 | 0.995370 | 0.972222 |

## Active Smoke/Inferno Intervals

- `8.5s` - `70.5s`, rows `125`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `52.0`, LSTM `0.6472`, XGBoost `0.8040`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `51.5`, LSTM `0.6438`, XGBoost `0.7867`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `51.0`, LSTM `0.6612`, XGBoost `0.7811`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `52.5`, LSTM `0.6537`, XGBoost `0.7674`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `50.0`, LSTM `0.7030`, XGBoost `0.8042`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `23.0`, LSTM `0.5600`, XGBoost `0.4805`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `23.5`, LSTM `0.5573`, XGBoost `0.4806`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `50.5`, LSTM `0.7059`, XGBoost `0.7816`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `53.5`, LSTM `0.8952`, XGBoost `0.9682`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `53.0`, LSTM `0.8999`, XGBoost `0.9685`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
