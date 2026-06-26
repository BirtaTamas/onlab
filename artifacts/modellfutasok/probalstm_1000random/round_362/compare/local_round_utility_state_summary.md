# Local Round Utility State Analysis

- csv_path: `processed_full/iem_cologne/iem-cologne-2025-ninjas-in-pyjamas-vs-natus-vincere-bo3-NyGFAYn592Hu0YfljTVG0N/ninjas-in-pyjamas-vs-natus-vincere-m3-train.csv`
- round_num: `17`
- rows: `230`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 230 | 1.000 | 0.833384 | 0.909623 | -0.076239 | 2 | 228 | 0.869565 | 0.991304 |
| active/recent utility | 230 | 1.000 | 0.833384 | 0.909623 | -0.076239 | 2 | 228 | 0.869565 | 0.991304 |
| strong utility action | 151 | 0.657 | 0.879603 | 0.940682 | -0.061079 | 2 | 149 | 0.966887 | 0.986755 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 151 | 0.657 | 0.879603 | 0.940682 | -0.061079 | 2 | 149 | 0.966887 | 0.986755 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 230 | 1.000 | 0.833384 | 0.909623 | -0.076239 | 2 | 228 | 0.869565 | 0.991304 |

## Active Smoke/Inferno Intervals

- `7.0s` - `42.5s`, rows `72`
- `52.5s` - `91.5s`, rows `79`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `85.0`, LSTM `0.7384`, XGBoost `0.9437`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `85.5`, LSTM `0.7679`, XGBoost `0.9422`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `86.0`, LSTM `0.5498`, XGBoost `0.7144`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `87.0`, LSTM `0.5600`, XGBoost `0.7160`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `87.5`, LSTM `0.5609`, XGBoost `0.7160`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `88.0`, LSTM `0.5707`, XGBoost `0.7160`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `84.5`, LSTM `0.8140`, XGBoost `0.9523`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `86.5`, LSTM `0.5794`, XGBoost `0.7160`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `91.5`, LSTM `0.4732`, XGBoost `0.6075`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `23.5`, LSTM `0.8397`, XGBoost `0.9726`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
