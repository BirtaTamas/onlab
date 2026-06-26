# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-virtuspro-vs-vitality-bo3-8Ft8K1evi_LZ8kW_kkrYdB/virtus-pro-vs-vitality-m1-train.csv`
- round_num: `9`
- rows: `228`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 228 | 1.000 | 0.656195 | 0.666750 | -0.010555 | 97 | 131 | 0.596491 | 0.491228 |
| active/recent utility | 228 | 1.000 | 0.656195 | 0.666750 | -0.010555 | 97 | 131 | 0.596491 | 0.491228 |
| strong utility action | 201 | 0.882 | 0.631171 | 0.642425 | -0.011254 | 90 | 111 | 0.567164 | 0.447761 |
| utility damage | 10 | 0.044 | 0.497757 | 0.490975 | 0.006781 | 7 | 3 | 0.600000 | 0.000000 |
| active smoke/inferno | 191 | 0.838 | 0.639215 | 0.649421 | -0.010206 | 89 | 102 | 0.581152 | 0.418848 |
| recent utility last 5s | 22 | 0.096 | 0.451475 | 0.490530 | -0.039055 | 3 | 19 | 0.181818 | 0.545455 |
| flash effect present | 228 | 1.000 | 0.656195 | 0.666750 | -0.010555 | 97 | 131 | 0.596491 | 0.491228 |

## Active Smoke/Inferno Intervals

- `6.5s` - `31.5s`, rows `51`
- `34.5s` - `104.0s`, rows `140`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `76.0`, LSTM `0.7456`, XGBoost `0.8805`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `75.5`, LSTM `0.7520`, XGBoost `0.8805`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `74.0`, LSTM `0.7671`, XGBoost `0.8890`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `74.5`, LSTM `0.7678`, XGBoost `0.8890`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `78.0`, LSTM `0.7322`, XGBoost `0.8519`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `62.0`, LSTM `0.2942`, XGBoost `0.4034`, closer `xgboost`, smoke `4`, inferno `1`, utility_damage `0.0`, recent_utility `1`
- seconds `75.0`, LSTM `0.7761`, XGBoost `0.8805`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `76.5`, LSTM `0.7823`, XGBoost `0.8805`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `61.5`, LSTM `0.3103`, XGBoost `0.4040`, closer `xgboost`, smoke `4`, inferno `1`, utility_damage `0.0`, recent_utility `1`
- seconds `77.5`, LSTM `0.7652`, XGBoost `0.8522`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
