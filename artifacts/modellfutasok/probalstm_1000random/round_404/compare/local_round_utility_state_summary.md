# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-imperial-vs-legacy-bo3-GRvbnL5Q4zT_JzAd-0AXgo/imperial-vs-legacy-m1-inferno.csv`
- round_num: `15`
- rows: `137`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 137 | 1.000 | 0.764815 | 0.825326 | -0.060510 | 12 | 125 | 1.000000 | 1.000000 |
| active/recent utility | 137 | 1.000 | 0.764815 | 0.825326 | -0.060510 | 12 | 125 | 1.000000 | 1.000000 |
| strong utility action | 115 | 0.839 | 0.791134 | 0.858450 | -0.067316 | 5 | 110 | 1.000000 | 1.000000 |
| utility damage | 20 | 0.146 | 0.870027 | 0.932311 | -0.062284 | 0 | 20 | 1.000000 | 1.000000 |
| active smoke/inferno | 115 | 0.839 | 0.791134 | 0.858450 | -0.067316 | 5 | 110 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 137 | 1.000 | 0.764815 | 0.825326 | -0.060510 | 12 | 125 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `11.0s` - `68.0s`, rows `115`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `38.0`, LSTM `0.6822`, XGBoost `0.8519`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `36.0`, LSTM `0.6866`, XGBoost `0.8518`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `35.5`, LSTM `0.6876`, XGBoost `0.8506`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `37.5`, LSTM `0.6981`, XGBoost `0.8496`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `36.5`, LSTM `0.7063`, XGBoost `0.8518`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `31.0`, LSTM `0.6896`, XGBoost `0.8349`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `37.0`, LSTM `0.7088`, XGBoost `0.8500`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `30.5`, LSTM `0.6970`, XGBoost `0.8341`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `31.5`, LSTM `0.6989`, XGBoost `0.8329`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `26.0`, LSTM `0.6979`, XGBoost `0.8291`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
