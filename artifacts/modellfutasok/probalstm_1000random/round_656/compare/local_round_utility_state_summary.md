# Local Round Utility State Analysis

- csv_path: `processed_full/blast_open_london_finals/blast-open-london-2025-finals-vitality-vs-g2-bo5-ieXHvClzA7f_aJ_85fPFqK/vitality-vs-g2-m5-train.csv`
- round_num: `8`
- rows: `134`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 134 | 1.000 | 0.628334 | 0.653630 | -0.025296 | 31 | 103 | 0.776119 | 0.843284 |
| active/recent utility | 134 | 1.000 | 0.628334 | 0.653630 | -0.025296 | 31 | 103 | 0.776119 | 0.843284 |
| strong utility action | 131 | 0.978 | 0.631236 | 0.656265 | -0.025029 | 31 | 100 | 0.770992 | 0.839695 |
| utility damage | 10 | 0.075 | 0.701839 | 0.731597 | -0.029759 | 3 | 7 | 1.000000 | 1.000000 |
| active smoke/inferno | 121 | 0.903 | 0.641958 | 0.665934 | -0.023976 | 31 | 90 | 0.793388 | 0.826446 |
| recent utility last 5s | 10 | 0.075 | 0.501495 | 0.539265 | -0.037770 | 0 | 10 | 0.500000 | 1.000000 |
| flash effect present | 134 | 1.000 | 0.628334 | 0.653630 | -0.025296 | 31 | 103 | 0.776119 | 0.843284 |

## Active Smoke/Inferno Intervals

- `6.5s` - `66.5s`, rows `121`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `37.5`, LSTM `0.6212`, XGBoost `0.7212`, closer `xgboost`, smoke `2`, inferno `2`, utility_damage `2.0`, recent_utility `0`
- seconds `38.5`, LSTM `0.6228`, XGBoost `0.7212`, closer `xgboost`, smoke `1`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `40.0`, LSTM `0.6230`, XGBoost `0.7212`, closer `xgboost`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `39.5`, LSTM `0.6241`, XGBoost `0.7212`, closer `xgboost`, smoke `1`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `39.0`, LSTM `0.6299`, XGBoost `0.7212`, closer `xgboost`, smoke `1`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `44.0`, LSTM `0.3919`, XGBoost `0.4815`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `16.5`, LSTM `0.4621`, XGBoost `0.5493`, closer `xgboost`, smoke `6`, inferno `2`, utility_damage `32.0`, recent_utility `0`
- seconds `38.0`, LSTM `0.6346`, XGBoost `0.7212`, closer `xgboost`, smoke `1`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `49.0`, LSTM `0.4239`, XGBoost `0.5073`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `43.5`, LSTM `0.3993`, XGBoost `0.4815`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
