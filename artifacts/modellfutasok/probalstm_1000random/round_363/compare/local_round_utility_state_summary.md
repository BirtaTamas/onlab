# Local Round Utility State Analysis

- csv_path: `processed_full/iem_cologne/iem-cologne-2025-ninjas-in-pyjamas-vs-natus-vincere-bo3-NyGFAYn592Hu0YfljTVG0N/ninjas-in-pyjamas-vs-natus-vincere-m3-train.csv`
- round_num: `3`
- rows: `123`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 123 | 1.000 | 0.921690 | 0.950964 | -0.029275 | 1 | 122 | 1.000000 | 1.000000 |
| active/recent utility | 123 | 1.000 | 0.921690 | 0.950964 | -0.029275 | 1 | 122 | 1.000000 | 1.000000 |
| strong utility action | 86 | 0.699 | 0.929514 | 0.958333 | -0.028819 | 1 | 85 | 1.000000 | 1.000000 |
| utility damage | 11 | 0.089 | 0.894936 | 0.932164 | -0.037228 | 0 | 11 | 1.000000 | 1.000000 |
| active smoke/inferno | 81 | 0.659 | 0.932395 | 0.961306 | -0.028911 | 1 | 80 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 123 | 1.000 | 0.921690 | 0.950964 | -0.029275 | 1 | 122 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `11.0s` - `32.5s`, rows `44`
- `43.0s` - `61.0s`, rows `37`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `12.0`, LSTM `0.9194`, XGBoost `0.9682`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `62.0`, recent_utility `0`
- seconds `11.5`, LSTM `0.8609`, XGBoost `0.9088`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `62.0`, recent_utility `0`
- seconds `11.0`, LSTM `0.8691`, XGBoost `0.9166`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `62.0`, recent_utility `0`
- seconds `12.5`, LSTM `0.9236`, XGBoost `0.9682`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `62.0`, recent_utility `0`
- seconds `13.0`, LSTM `0.9270`, XGBoost `0.9706`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `62.0`, recent_utility `0`
- seconds `48.5`, LSTM `0.9033`, XGBoost `0.9445`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `13.5`, LSTM `0.9302`, XGBoost `0.9706`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `44.0`, recent_utility `0`
- seconds `52.5`, LSTM `0.9038`, XGBoost `0.9440`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `15.0`, LSTM `0.9363`, XGBoost `0.9750`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `14.0`, LSTM `0.9323`, XGBoost `0.9711`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
