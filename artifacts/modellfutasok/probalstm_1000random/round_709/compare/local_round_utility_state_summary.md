# Local Round Utility State Analysis

- csv_path: `processed_full/blast_open_lisbon/blast-open-lisbon-2025-faze-vs-virtuspro-bo3-YK5CkfojMMEdQ3U0_CVA2l/faze-vs-virtuspro-m1-anubis.csv`
- round_num: `23`
- rows: `255`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 255 | 1.000 | 0.427691 | 0.392392 | 0.035299 | 57 | 198 | 0.435294 | 0.937255 |
| active/recent utility | 255 | 1.000 | 0.427691 | 0.392392 | 0.035299 | 57 | 198 | 0.435294 | 0.937255 |
| strong utility action | 218 | 0.855 | 0.435417 | 0.392926 | 0.042491 | 36 | 182 | 0.412844 | 0.990826 |
| utility damage | 18 | 0.071 | 0.403391 | 0.336615 | 0.066776 | 0 | 18 | 0.555556 | 1.000000 |
| active smoke/inferno | 218 | 0.855 | 0.435417 | 0.392926 | 0.042491 | 36 | 182 | 0.412844 | 0.990826 |
| recent utility last 5s | 10 | 0.039 | 0.527779 | 0.480687 | 0.047092 | 0 | 10 | 0.000000 | 1.000000 |
| flash effect present | 255 | 1.000 | 0.427691 | 0.392392 | 0.035299 | 57 | 198 | 0.435294 | 0.937255 |

## Active Smoke/Inferno Intervals

- `6.5s` - `40.5s`, rows `69`
- `42.0s` - `95.5s`, rows `108`
- `107.0s` - `127.0s`, rows `41`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `112.0`, LSTM `0.1281`, XGBoost `0.2679`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `112.5`, LSTM `0.1261`, XGBoost `0.2654`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `53.0`, LSTM `0.4568`, XGBoost `0.3185`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `55.0`, LSTM `0.3964`, XGBoost `0.2588`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `113.0`, LSTM `0.1294`, XGBoost `0.2654`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `109.5`, LSTM `0.1023`, XGBoost `0.2341`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `111.5`, LSTM `0.1244`, XGBoost `0.2544`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `64.5`, LSTM `0.3802`, XGBoost `0.2561`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `114.0`, LSTM `0.1411`, XGBoost `0.2647`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `113.5`, LSTM `0.1418`, XGBoost `0.2627`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
