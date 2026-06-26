# Local Round Utility State Analysis

- csv_path: `processed_full/blast_open_london_finals/blast-open-london-2025-finals-faze-vs-g2-bo3-ldI7_iFRuThMOXF8zIbBwX/faze-vs-g2-m1-inferno.csv`
- round_num: `1`
- rows: `209`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 209 | 1.000 | 0.320468 | 0.360319 | -0.039851 | 162 | 47 | 0.741627 | 0.483254 |
| active/recent utility | 155 | 0.742 | 0.261624 | 0.314190 | -0.052567 | 138 | 17 | 0.916129 | 0.625806 |
| strong utility action | 69 | 0.330 | 0.203978 | 0.226728 | -0.022751 | 53 | 16 | 0.956522 | 0.811594 |
| utility damage | 10 | 0.048 | 0.391208 | 0.381465 | 0.009742 | 6 | 4 | 0.700000 | 0.600000 |
| active smoke/inferno | 64 | 0.306 | 0.195891 | 0.221615 | -0.025724 | 51 | 13 | 0.953125 | 0.796875 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 155 | 0.742 | 0.261624 | 0.314190 | -0.052567 | 138 | 17 | 0.916129 | 0.625806 |

## Active Smoke/Inferno Intervals

- `39.5s` - `46.0s`, rows `14`
- `74.5s` - `99.0s`, rows `50`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `78.5`, LSTM `0.3470`, XGBoost `0.2346`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `78.0`, LSTM `0.3194`, XGBoost `0.2339`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `44.0`, LSTM `0.4516`, XGBoost `0.5358`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `37.0`, LSTM `0.3115`, XGBoost `0.3924`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `136.0`, recent_utility `0`
- seconds `39.5`, LSTM `0.3598`, XGBoost `0.2801`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `136.0`, recent_utility `0`
- seconds `44.5`, LSTM `0.4386`, XGBoost `0.5180`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `90.0`, LSTM `0.0476`, XGBoost `0.1242`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `89.5`, LSTM `0.0479`, XGBoost `0.1242`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `89.0`, LSTM `0.0481`, XGBoost `0.1242`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `87.0`, LSTM `0.0487`, XGBoost `0.1247`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
