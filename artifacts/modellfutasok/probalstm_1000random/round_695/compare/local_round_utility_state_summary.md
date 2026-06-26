# Local Round Utility State Analysis

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-furia-vs-g2-bo3-_Q5oHsnKyowoqEbJh5o3f8/furia-vs-g2-m2-overpass.csv`
- round_num: `24`
- rows: `216`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 216 | 1.000 | 0.677969 | 0.694774 | -0.016805 | 66 | 150 | 1.000000 | 0.986111 |
| active/recent utility | 216 | 1.000 | 0.677969 | 0.694774 | -0.016805 | 66 | 150 | 1.000000 | 0.986111 |
| strong utility action | 209 | 0.968 | 0.679931 | 0.697475 | -0.017544 | 62 | 147 | 1.000000 | 0.985646 |
| utility damage | 40 | 0.185 | 0.855831 | 0.875256 | -0.019425 | 11 | 29 | 1.000000 | 0.925000 |
| active smoke/inferno | 199 | 0.921 | 0.681380 | 0.702396 | -0.021016 | 52 | 147 | 1.000000 | 0.984925 |
| recent utility last 5s | 10 | 0.046 | 0.651089 | 0.599537 | 0.051552 | 10 | 0 | 1.000000 | 1.000000 |
| flash effect present | 216 | 1.000 | 0.677969 | 0.694774 | -0.016805 | 66 | 150 | 1.000000 | 0.986111 |

## Active Smoke/Inferno Intervals

- `8.5s` - `107.5s`, rows `199`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `107.0`, LSTM `0.5913`, XGBoost `0.8112`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `107.5`, LSTM `0.5769`, XGBoost `0.7947`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `72.0`, LSTM `0.6870`, XGBoost `0.4758`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `25.0`, recent_utility `0`
- seconds `72.5`, LSTM `0.6844`, XGBoost `0.4763`, closer `lstm`, smoke `2`, inferno `3`, utility_damage `25.0`, recent_utility `0`
- seconds `73.0`, LSTM `0.6850`, XGBoost `0.4857`, closer `lstm`, smoke `2`, inferno `3`, utility_damage `25.0`, recent_utility `0`
- seconds `106.0`, LSTM `0.6264`, XGBoost `0.8060`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `13.0`, recent_utility `0`
- seconds `106.5`, LSTM `0.6382`, XGBoost `0.8060`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `105.5`, LSTM `0.6430`, XGBoost `0.7951`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `13.0`, recent_utility `0`
- seconds `105.0`, LSTM `0.6511`, XGBoost `0.7926`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `13.0`, recent_utility `0`
- seconds `71.5`, LSTM `0.6530`, XGBoost `0.5508`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `14.0`, recent_utility `0`
