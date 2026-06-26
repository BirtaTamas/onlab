# Local Round Utility State Analysis

- csv_path: `processed_full/blast_open_london_finals/blast-open-london-2025-finals-mouz-vs-m80-bo3-v7WxfaSDQDAUAgkS_SwEt2/mouz-vs-m80-m3-inferno.csv`
- round_num: `2`
- rows: `165`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 165 | 1.000 | 0.725015 | 0.748430 | -0.023415 | 74 | 91 | 1.000000 | 1.000000 |
| active/recent utility | 165 | 1.000 | 0.725015 | 0.748430 | -0.023415 | 74 | 91 | 1.000000 | 1.000000 |
| strong utility action | 153 | 0.927 | 0.729272 | 0.753934 | -0.024662 | 68 | 85 | 1.000000 | 1.000000 |
| utility damage | 34 | 0.206 | 0.809960 | 0.800829 | 0.009132 | 20 | 14 | 1.000000 | 1.000000 |
| active smoke/inferno | 140 | 0.848 | 0.736985 | 0.766257 | -0.029272 | 56 | 84 | 1.000000 | 1.000000 |
| recent utility last 5s | 10 | 0.061 | 0.641594 | 0.619295 | 0.022299 | 9 | 1 | 1.000000 | 1.000000 |
| flash effect present | 165 | 1.000 | 0.725015 | 0.748430 | -0.023415 | 74 | 91 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `11.5s` - `81.0s`, rows `140`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `52.5`, LSTM `0.6364`, XGBoost `0.7562`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `51.0`, LSTM `0.6403`, XGBoost `0.7566`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `64.0`, LSTM `0.6310`, XGBoost `0.7450`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `74.5`, LSTM `0.8172`, XGBoost `0.7060`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `2.0`, recent_utility `0`
- seconds `58.5`, LSTM `0.6306`, XGBoost `0.7406`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `58.0`, LSTM `0.6315`, XGBoost `0.7406`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `50.5`, LSTM `0.6481`, XGBoost `0.7563`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `50.0`, LSTM `0.6509`, XGBoost `0.7563`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `59.0`, LSTM `0.6377`, XGBoost `0.7406`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `49.5`, LSTM `0.6540`, XGBoost `0.7563`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
