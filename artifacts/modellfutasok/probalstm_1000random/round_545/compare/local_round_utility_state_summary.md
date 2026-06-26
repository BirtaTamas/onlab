# Local Round Utility State Analysis

- csv_path: `processed_full/blast_rivals_season_2/blast-rivals-2025-season-2-furia-vs-falcons-bo5-L7CZVGSHd1AqjKPyYU04lA/furia-vs-falcons-m1-inferno.csv`
- round_num: `2`
- rows: `222`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 222 | 1.000 | 0.629415 | 0.689672 | -0.060257 | 4 | 218 | 1.000000 | 1.000000 |
| active/recent utility | 222 | 1.000 | 0.629415 | 0.689672 | -0.060257 | 4 | 218 | 1.000000 | 1.000000 |
| strong utility action | 203 | 0.914 | 0.637931 | 0.696238 | -0.058307 | 4 | 199 | 1.000000 | 1.000000 |
| utility damage | 10 | 0.045 | 0.581123 | 0.615059 | -0.033936 | 1 | 9 | 1.000000 | 1.000000 |
| active smoke/inferno | 195 | 0.878 | 0.641448 | 0.699458 | -0.058010 | 4 | 191 | 1.000000 | 1.000000 |
| recent utility last 5s | 21 | 0.095 | 0.561302 | 0.618131 | -0.056829 | 0 | 21 | 1.000000 | 1.000000 |
| flash effect present | 222 | 1.000 | 0.629415 | 0.689672 | -0.060257 | 4 | 218 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `10.0s` - `53.0s`, rows `87`
- `57.0s` - `110.5s`, rows `108`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `48.5`, LSTM `0.5058`, XGBoost `0.6189`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `49.0`, LSTM `0.5093`, XGBoost `0.6192`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `48.0`, LSTM `0.5094`, XGBoost `0.6189`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `49.5`, LSTM `0.5115`, XGBoost `0.6192`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `47.5`, LSTM `0.5160`, XGBoost `0.6189`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `47.0`, LSTM `0.5227`, XGBoost `0.6189`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `110.0`, LSTM `0.8875`, XGBoost `0.9808`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `51.0`, LSTM `0.5272`, XGBoost `0.6203`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `87.5`, LSTM `0.5281`, XGBoost `0.6205`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `17.0`, LSTM `0.5223`, XGBoost `0.6141`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
