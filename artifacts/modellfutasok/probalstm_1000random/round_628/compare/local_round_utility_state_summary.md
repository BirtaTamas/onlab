# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-virtuspro-vs-vitality-bo3-8Ft8K1evi_LZ8kW_kkrYdB/virtus-pro-vs-vitality-m1-train.csv`
- round_num: `14`
- rows: `257`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 257 | 1.000 | 0.439769 | 0.448144 | -0.008375 | 174 | 83 | 0.264591 | 0.365759 |
| active/recent utility | 257 | 1.000 | 0.439769 | 0.448144 | -0.008375 | 174 | 83 | 0.264591 | 0.365759 |
| strong utility action | 212 | 0.825 | 0.478287 | 0.485525 | -0.007238 | 152 | 60 | 0.188679 | 0.311321 |
| utility damage | 10 | 0.039 | 0.595776 | 0.618738 | -0.022962 | 9 | 1 | 0.000000 | 0.000000 |
| active smoke/inferno | 212 | 0.825 | 0.478287 | 0.485525 | -0.007238 | 152 | 60 | 0.188679 | 0.311321 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 257 | 1.000 | 0.439769 | 0.448144 | -0.008375 | 174 | 83 | 0.264591 | 0.365759 |

## Active Smoke/Inferno Intervals

- `8.5s` - `114.0s`, rows `212`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `87.5`, LSTM `0.5459`, XGBoost `0.3640`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `16.0`, recent_utility `0`
- seconds `89.0`, LSTM `0.6218`, XGBoost `0.4627`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `6.0`, recent_utility `0`
- seconds `90.5`, LSTM `0.6194`, XGBoost `0.4614`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `14.0`, recent_utility `0`
- seconds `95.5`, LSTM `0.5210`, XGBoost `0.3756`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `48.0`, recent_utility `0`
- seconds `90.0`, LSTM `0.6069`, XGBoost `0.4619`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `14.0`, recent_utility `0`
- seconds `96.5`, LSTM `0.5099`, XGBoost `0.3750`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `24.0`, recent_utility `0`
- seconds `91.0`, LSTM `0.5486`, XGBoost `0.4154`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `22.0`, recent_utility `0`
- seconds `89.5`, LSTM `0.5934`, XGBoost `0.4629`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `14.0`, recent_utility `0`
- seconds `96.0`, LSTM `0.5049`, XGBoost `0.3756`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `40.0`, recent_utility `0`
- seconds `91.5`, LSTM `0.5434`, XGBoost `0.4168`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `38.0`, recent_utility `0`
