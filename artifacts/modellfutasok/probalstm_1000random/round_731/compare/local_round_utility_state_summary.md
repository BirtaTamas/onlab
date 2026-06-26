# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-mouz-vs-vitality-bo3-pYxpz34IEN-t8y4DgB-MSD/mouz-vs-vitality-m3-train.csv`
- round_num: `17`
- rows: `191`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 191 | 1.000 | 0.838848 | 0.871339 | -0.032491 | 8 | 183 | 1.000000 | 1.000000 |
| active/recent utility | 191 | 1.000 | 0.838848 | 0.871339 | -0.032491 | 8 | 183 | 1.000000 | 1.000000 |
| strong utility action | 145 | 0.759 | 0.816512 | 0.849944 | -0.033432 | 8 | 137 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 145 | 0.759 | 0.816512 | 0.849944 | -0.033432 | 8 | 137 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 191 | 1.000 | 0.838848 | 0.871339 | -0.032491 | 8 | 183 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `8.5s` - `80.5s`, rows `145`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `32.5`, LSTM `0.7454`, XGBoost `0.8453`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `33.5`, LSTM `0.7495`, XGBoost `0.8456`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `32.0`, LSTM `0.7555`, XGBoost `0.8453`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `31.5`, LSTM `0.7432`, XGBoost `0.8295`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `34.0`, LSTM `0.7448`, XGBoost `0.8299`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `33.0`, LSTM `0.7607`, XGBoost `0.8454`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `30.5`, LSTM `0.7457`, XGBoost `0.8299`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `30.0`, LSTM `0.7506`, XGBoost `0.8299`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `35.5`, LSTM `0.7612`, XGBoost `0.8404`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `31.0`, LSTM `0.7508`, XGBoost `0.8297`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
