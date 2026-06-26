# Local Round Utility State Analysis

- csv_path: `processed_full/iem_katowice/iem-katowice-2025-virtuspro-vs-spirit-bo3-KJqZR5yNeHXaNsc7MGaDWB/virtus-pro-vs-spirit-m1-train.csv`
- round_num: `7`
- rows: `133`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 133 | 1.000 | 0.741715 | 0.788801 | -0.047087 | 4 | 129 | 1.000000 | 1.000000 |
| active/recent utility | 133 | 1.000 | 0.741715 | 0.788801 | -0.047087 | 4 | 129 | 1.000000 | 1.000000 |
| strong utility action | 107 | 0.805 | 0.739416 | 0.790316 | -0.050900 | 4 | 103 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 92 | 0.692 | 0.744545 | 0.781294 | -0.036749 | 4 | 88 | 1.000000 | 1.000000 |
| recent utility last 5s | 15 | 0.113 | 0.707958 | 0.845653 | -0.137695 | 0 | 15 | 1.000000 | 1.000000 |
| flash effect present | 133 | 1.000 | 0.741715 | 0.788801 | -0.047087 | 4 | 129 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `8.5s` - `13.5s`, rows `11`
- `15.5s` - `20.5s`, rows `11`
- `31.5s` - `66.0s`, rows `70`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `4.0`, LSTM `0.6640`, XGBoost `0.8449`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `4`
- seconds `5.0`, LSTM `0.6701`, XGBoost `0.8448`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `4`
- seconds `4.5`, LSTM `0.6699`, XGBoost `0.8445`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `4`
- seconds `3.5`, LSTM `0.6754`, XGBoost `0.8461`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `4`
- seconds `7.5`, LSTM `0.6785`, XGBoost `0.8443`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `5.5`, LSTM `0.6863`, XGBoost `0.8448`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `2`
- seconds `6.0`, LSTM `0.6925`, XGBoost `0.8448`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `2`
- seconds `3.0`, LSTM `0.6944`, XGBoost `0.8443`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `4`
- seconds `7.0`, LSTM `0.7005`, XGBoost `0.8443`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `2`
- seconds `6.5`, LSTM `0.7031`, XGBoost `0.8443`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `2`
