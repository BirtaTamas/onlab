# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-b8-vs-wildcard-bo3-EO1cCePneo0X8r6rxB_BMC/b8-vs-wildcard-m3-inferno.csv`
- round_num: `4`
- rows: `218`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 218 | 1.000 | 0.872086 | 0.880032 | -0.007946 | 49 | 169 | 1.000000 | 0.917431 |
| active/recent utility | 218 | 1.000 | 0.872086 | 0.880032 | -0.007946 | 49 | 169 | 1.000000 | 0.917431 |
| strong utility action | 184 | 0.844 | 0.897442 | 0.910087 | -0.012645 | 30 | 154 | 1.000000 | 1.000000 |
| utility damage | 21 | 0.096 | 0.737700 | 0.716061 | 0.021639 | 13 | 8 | 1.000000 | 1.000000 |
| active smoke/inferno | 183 | 0.839 | 0.897068 | 0.909710 | -0.012643 | 30 | 153 | 1.000000 | 1.000000 |
| recent utility last 5s | 10 | 0.046 | 0.950302 | 0.984025 | -0.033723 | 0 | 10 | 1.000000 | 1.000000 |
| flash effect present | 218 | 1.000 | 0.872086 | 0.880032 | -0.007946 | 49 | 169 | 1.000000 | 0.917431 |

## Active Smoke/Inferno Intervals

- `9.5s` - `33.0s`, rows `48`
- `34.0s` - `69.0s`, rows `71`
- `77.0s` - `108.5s`, rows `64`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `19.5`, LSTM `0.5980`, XGBoost `0.5083`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `19.0`, LSTM `0.5925`, XGBoost `0.5083`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `17.5`, LSTM `0.5776`, XGBoost `0.5083`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `18.0`, LSTM `0.5769`, XGBoost `0.5083`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `17.0`, LSTM `0.5765`, XGBoost `0.5083`, closer `lstm`, smoke `1`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `18.5`, LSTM `0.5713`, XGBoost `0.5083`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `16.0`, LSTM `0.5683`, XGBoost `0.5061`, closer `lstm`, smoke `1`, inferno `3`, utility_damage `3.0`, recent_utility `0`
- seconds `21.0`, LSTM `0.6580`, XGBoost `0.7189`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `20.0`, LSTM `0.6165`, XGBoost `0.5576`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `9.5`, LSTM `0.5600`, XGBoost `0.5025`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
