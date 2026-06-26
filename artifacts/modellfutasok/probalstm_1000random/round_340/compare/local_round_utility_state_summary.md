# Local Round Utility State Analysis

- csv_path: `processed_full/asian_champions_league/hero-esports-asian-champions-league-2025-lynn-vision-vs-ninja-bo3-zpPbzx1DSQhVYC3-qoelpd/lynn-vision-vs-ninja-m2-inferno.csv`
- round_num: `16`
- rows: `215`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 215 | 1.000 | 0.665273 | 0.676272 | -0.010999 | 102 | 113 | 0.795349 | 0.511628 |
| active/recent utility | 215 | 1.000 | 0.665273 | 0.676272 | -0.010999 | 102 | 113 | 0.795349 | 0.511628 |
| strong utility action | 161 | 0.749 | 0.649935 | 0.666727 | -0.016791 | 72 | 89 | 0.832298 | 0.534161 |
| utility damage | 20 | 0.093 | 0.787143 | 0.808418 | -0.021276 | 4 | 16 | 1.000000 | 1.000000 |
| active smoke/inferno | 161 | 0.749 | 0.649935 | 0.666727 | -0.016791 | 72 | 89 | 0.832298 | 0.534161 |
| recent utility last 5s | 10 | 0.047 | 0.798899 | 0.911004 | -0.112104 | 0 | 10 | 1.000000 | 1.000000 |
| flash effect present | 215 | 1.000 | 0.665273 | 0.676272 | -0.010999 | 102 | 113 | 0.795349 | 0.511628 |

## Active Smoke/Inferno Intervals

- `6.5s` - `39.5s`, rows `67`
- `48.5s` - `89.5s`, rows `83`
- `91.5s` - `96.5s`, rows `11`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `83.0`, LSTM `0.7491`, XGBoost `0.9083`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `80.0`, LSTM `0.7655`, XGBoost `0.9196`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `8.0`, recent_utility `0`
- seconds `81.0`, LSTM `0.7733`, XGBoost `0.9238`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `82.5`, LSTM `0.7579`, XGBoost `0.9053`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `79.5`, LSTM `0.7771`, XGBoost `0.9170`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `16.0`, recent_utility `0`
- seconds `78.5`, LSTM `0.7756`, XGBoost `0.9131`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `16.0`, recent_utility `0`
- seconds `82.0`, LSTM `0.7677`, XGBoost `0.8988`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `80.5`, LSTM `0.7914`, XGBoost `0.9214`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `81.5`, LSTM `0.7630`, XGBoost `0.8921`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `85.0`, LSTM `0.7875`, XGBoost `0.9156`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `1`
