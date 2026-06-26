# Local Round Utility State Analysis

- csv_path: `processed_full/blast_rivals_season_2/blast-rivals-2025-season-2-vitality-vs-tyloo-bo3-WTYOidpO-mHqROoLZlA7Li/vitality-vs-tyloo-m1-overpass.csv`
- round_num: `3`
- rows: `230`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 230 | 1.000 | 0.767451 | 0.886026 | -0.118575 | 6 | 224 | 1.000000 | 1.000000 |
| active/recent utility | 230 | 1.000 | 0.767451 | 0.886026 | -0.118575 | 6 | 224 | 1.000000 | 1.000000 |
| strong utility action | 198 | 0.861 | 0.773398 | 0.889077 | -0.115679 | 5 | 193 | 1.000000 | 1.000000 |
| utility damage | 11 | 0.048 | 0.747823 | 0.822994 | -0.075171 | 0 | 11 | 1.000000 | 1.000000 |
| active smoke/inferno | 191 | 0.830 | 0.773378 | 0.894307 | -0.120929 | 0 | 191 | 1.000000 | 1.000000 |
| recent utility last 5s | 16 | 0.070 | 0.753594 | 0.789715 | -0.036121 | 5 | 11 | 1.000000 | 1.000000 |
| flash effect present | 230 | 1.000 | 0.767451 | 0.886026 | -0.118575 | 6 | 224 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `6.0s` - `27.5s`, rows `44`
- `35.0s` - `77.5s`, rows `86`
- `84.5s` - `114.5s`, rows `61`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `91.5`, LSTM `0.6531`, XGBoost `0.9260`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `91.0`, LSTM `0.6697`, XGBoost `0.9260`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `92.0`, LSTM `0.6733`, XGBoost `0.9257`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `64.5`, LSTM `0.6961`, XGBoost `0.9361`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `44.5`, LSTM `0.5879`, XGBoost `0.8266`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `66.0`, LSTM `0.6992`, XGBoost `0.9314`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `110.0`, LSTM `0.7265`, XGBoost `0.9531`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `69.0`, LSTM `0.7055`, XGBoost `0.9309`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `68.5`, LSTM `0.7060`, XGBoost `0.9309`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `65.5`, LSTM `0.7068`, XGBoost `0.9314`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
