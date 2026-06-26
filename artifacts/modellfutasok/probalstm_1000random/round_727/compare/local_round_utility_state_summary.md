# Local Round Utility State Analysis

- csv_path: `processed_full/blast_bounty_season_1/blast-bounty-2025-season-1-eternal-fire-vs-fluxo-bo3-Kqy3ohBVu1ANumI6Qdn26R/eternal-fire-vs-fluxo-m2-dust2.csv`
- round_num: `15`
- rows: `284`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 284 | 1.000 | 0.403556 | 0.334733 | 0.068822 | 91 | 193 | 0.376761 | 0.549296 |
| active/recent utility | 284 | 1.000 | 0.403556 | 0.334733 | 0.068822 | 91 | 193 | 0.376761 | 0.549296 |
| strong utility action | 267 | 0.940 | 0.414396 | 0.343444 | 0.070952 | 89 | 178 | 0.363296 | 0.520599 |
| utility damage | 30 | 0.106 | 0.683716 | 0.553647 | 0.130069 | 0 | 30 | 0.000000 | 0.000000 |
| active smoke/inferno | 250 | 0.880 | 0.419916 | 0.347640 | 0.072276 | 84 | 166 | 0.360000 | 0.488000 |
| recent utility last 5s | 30 | 0.106 | 0.198740 | 0.175464 | 0.023276 | 18 | 12 | 0.666667 | 1.000000 |
| flash effect present | 284 | 1.000 | 0.403556 | 0.334733 | 0.068822 | 91 | 193 | 0.376761 | 0.549296 |

## Active Smoke/Inferno Intervals

- `8.5s` - `103.0s`, rows `190`
- `105.0s` - `134.5s`, rows `60`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `50.0`, LSTM `0.7259`, XGBoost `0.5283`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `50.5`, LSTM `0.7118`, XGBoost `0.5283`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `49.5`, LSTM `0.7110`, XGBoost `0.5283`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `48.5`, LSTM `0.7052`, XGBoost `0.5260`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `35.5`, LSTM `0.7024`, XGBoost `0.5247`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `32.0`, recent_utility `0`
- seconds `51.5`, LSTM `0.7057`, XGBoost `0.5285`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `49.0`, LSTM `0.7018`, XGBoost `0.5260`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `48.0`, LSTM `0.6995`, XGBoost `0.5260`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `51.0`, LSTM `0.7005`, XGBoost `0.5283`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `88.0`, LSTM `0.6830`, XGBoost `0.5150`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
