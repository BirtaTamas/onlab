# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-imperial-vs-legacy-bo3-GRvbnL5Q4zT_JzAd-0AXgo/imperial-vs-legacy-m3-mirage.csv`
- round_num: `2`
- rows: `127`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 127 | 1.000 | 0.926290 | 0.967639 | -0.041350 | 22 | 105 | 1.000000 | 1.000000 |
| active/recent utility | 127 | 1.000 | 0.926290 | 0.967639 | -0.041350 | 22 | 105 | 1.000000 | 1.000000 |
| strong utility action | 86 | 0.677 | 0.919971 | 0.971155 | -0.051184 | 3 | 83 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 86 | 0.677 | 0.919971 | 0.971155 | -0.051184 | 3 | 83 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 127 | 1.000 | 0.926290 | 0.967639 | -0.041350 | 22 | 105 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `6.5s` - `11.5s`, rows `11`
- `14.5s` - `51.5s`, rows `75`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `40.5`, LSTM `0.8671`, XGBoost `0.9670`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `41.0`, LSTM `0.8726`, XGBoost `0.9677`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `41.5`, LSTM `0.8770`, XGBoost `0.9677`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `7.5`, LSTM `0.8874`, XGBoost `0.9748`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `6.5`, LSTM `0.8873`, XGBoost `0.9747`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `7.0`, LSTM `0.8876`, XGBoost `0.9747`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `10.5`, LSTM `0.8914`, XGBoost `0.9755`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `42.0`, LSTM `0.8844`, XGBoost `0.9677`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `35.5`, LSTM `0.8824`, XGBoost `0.9652`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `14.5`, LSTM `0.8950`, XGBoost `0.9758`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
