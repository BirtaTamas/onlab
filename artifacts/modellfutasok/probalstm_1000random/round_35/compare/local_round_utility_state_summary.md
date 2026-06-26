# Local Round Utility State Analysis

- csv_path: `processed_full/blast_open_lisbon/blast-open-lisbon-2025-mouz-vs-falcons-bo3-xBECUqZMcQ8GCwi-GUyz8e/mouz-vs-falcons-m1-train.csv`
- round_num: `4`
- rows: `220`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 220 | 1.000 | 0.269644 | 0.280579 | -0.010935 | 171 | 49 | 0.740909 | 0.568182 |
| active/recent utility | 220 | 1.000 | 0.269644 | 0.280579 | -0.010935 | 171 | 49 | 0.740909 | 0.568182 |
| strong utility action | 138 | 0.627 | 0.372504 | 0.379887 | -0.007383 | 90 | 48 | 0.594203 | 0.434783 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 138 | 0.627 | 0.372504 | 0.379887 | -0.007383 | 90 | 48 | 0.594203 | 0.434783 |
| recent utility last 5s | 10 | 0.045 | 0.488013 | 0.499912 | -0.011899 | 7 | 3 | 0.800000 | 0.700000 |
| flash effect present | 220 | 1.000 | 0.269644 | 0.280579 | -0.010935 | 171 | 49 | 0.740909 | 0.568182 |

## Active Smoke/Inferno Intervals

- `8.0s` - `50.0s`, rows `85`
- `51.0s` - `77.0s`, rows `53`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `9.5`, LSTM `0.4432`, XGBoost `0.5491`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `10.0`, LSTM `0.4500`, XGBoost `0.5491`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `9.0`, LSTM `0.4504`, XGBoost `0.5480`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `61.0`, LSTM `0.0815`, XGBoost `0.1765`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `59.5`, LSTM `0.2641`, XGBoost `0.3577`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `60.5`, LSTM `0.0871`, XGBoost `0.1765`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `8.0`, LSTM `0.4610`, XGBoost `0.5476`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `10.5`, LSTM `0.4536`, XGBoost `0.5311`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `21.0`, recent_utility `0`
- seconds `55.5`, LSTM `0.5037`, XGBoost `0.4266`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `11.0`, LSTM `0.4595`, XGBoost `0.5301`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `21.0`, recent_utility `0`
