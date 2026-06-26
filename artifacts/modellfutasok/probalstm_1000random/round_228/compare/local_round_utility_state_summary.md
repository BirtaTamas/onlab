# Local Round Utility State Analysis

- csv_path: `processed_full/blast_open_london_finals/blast-open-london-2025-finals-vitality-vs-g2-bo5-ieXHvClzA7f_aJ_85fPFqK/vitality-vs-g2-m5-train.csv`
- round_num: `11`
- rows: `255`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 255 | 1.000 | 0.540640 | 0.698165 | -0.157525 | 9 | 246 | 0.635294 | 0.878431 |
| active/recent utility | 255 | 1.000 | 0.540640 | 0.698165 | -0.157525 | 9 | 246 | 0.635294 | 0.878431 |
| strong utility action | 208 | 0.816 | 0.524609 | 0.683437 | -0.158828 | 9 | 199 | 0.653846 | 0.884615 |
| utility damage | 10 | 0.039 | 0.509853 | 0.677265 | -0.167412 | 0 | 10 | 0.700000 | 1.000000 |
| active smoke/inferno | 197 | 0.773 | 0.531761 | 0.692325 | -0.160564 | 9 | 188 | 0.690355 | 0.878173 |
| recent utility last 5s | 13 | 0.051 | 0.395245 | 0.526515 | -0.131270 | 0 | 13 | 0.000000 | 1.000000 |
| flash effect present | 255 | 1.000 | 0.540640 | 0.698165 | -0.157525 | 9 | 246 | 0.635294 | 0.878431 |

## Active Smoke/Inferno Intervals

- `6.0s` - `104.0s`, rows `197`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `102.5`, LSTM `0.1178`, XGBoost `0.4882`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `101.5`, LSTM `0.1114`, XGBoost `0.4805`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `102.0`, LSTM `0.1154`, XGBoost `0.4788`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `101.0`, LSTM `0.1277`, XGBoost `0.4805`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `100.5`, LSTM `0.1284`, XGBoost `0.4786`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `103.0`, LSTM `0.1451`, XGBoost `0.4938`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `100.0`, LSTM `0.1346`, XGBoost `0.4626`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `98.5`, LSTM `0.1460`, XGBoost `0.4718`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `99.5`, LSTM `0.1430`, XGBoost `0.4585`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `99.0`, LSTM `0.1546`, XGBoost `0.4569`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
