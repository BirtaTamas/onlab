# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-vitality-vs-mouz-bo3-Ko5VJMvyF1OsCx2TbVU9pb/vitality-vs-mouz-m1-inferno.csv`
- round_num: `4`
- rows: `199`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 199 | 1.000 | 0.502364 | 0.556905 | -0.054540 | 48 | 151 | 0.457286 | 0.391960 |
| active/recent utility | 199 | 1.000 | 0.502364 | 0.556905 | -0.054540 | 48 | 151 | 0.457286 | 0.391960 |
| strong utility action | 177 | 0.889 | 0.466421 | 0.524315 | -0.057894 | 42 | 135 | 0.412429 | 0.355932 |
| utility damage | 20 | 0.101 | 0.299584 | 0.332432 | -0.032849 | 4 | 16 | 0.150000 | 0.000000 |
| active smoke/inferno | 167 | 0.839 | 0.465400 | 0.526779 | -0.061379 | 39 | 128 | 0.431138 | 0.377246 |
| recent utility last 5s | 10 | 0.050 | 0.483467 | 0.483168 | 0.000299 | 3 | 7 | 0.100000 | 0.000000 |
| flash effect present | 199 | 1.000 | 0.502364 | 0.556905 | -0.054540 | 48 | 151 | 0.457286 | 0.391960 |

## Active Smoke/Inferno Intervals

- `8.5s` - `91.5s`, rows `167`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `65.5`, LSTM `0.7197`, XGBoost `0.9475`, closer `xgboost`, smoke `4`, inferno `1`, utility_damage `13.0`, recent_utility `0`
- seconds `78.0`, LSTM `0.6985`, XGBoost `0.9213`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `79.5`, LSTM `0.7209`, XGBoost `0.9401`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `77.5`, LSTM `0.7085`, XGBoost `0.9213`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `73.5`, LSTM `0.7136`, XGBoost `0.9239`, closer `xgboost`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `65.0`, LSTM `0.7380`, XGBoost `0.9472`, closer `xgboost`, smoke `4`, inferno `1`, utility_damage `13.0`, recent_utility `0`
- seconds `74.0`, LSTM `0.7167`, XGBoost `0.9239`, closer `xgboost`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `77.0`, LSTM `0.7145`, XGBoost `0.9213`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `78.5`, LSTM `0.7150`, XGBoost `0.9213`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `72.5`, LSTM `0.7178`, XGBoost `0.9239`, closer `xgboost`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
