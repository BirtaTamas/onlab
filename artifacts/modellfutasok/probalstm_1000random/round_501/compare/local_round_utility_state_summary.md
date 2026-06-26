# Local Round Utility State Analysis

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-furia-vs-g2-bo3-_Q5oHsnKyowoqEbJh5o3f8/furia-vs-g2-m2-overpass.csv`
- round_num: `1`
- rows: `208`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 208 | 1.000 | 0.732495 | 0.799430 | -0.066935 | 7 | 201 | 0.774038 | 1.000000 |
| active/recent utility | 208 | 1.000 | 0.732495 | 0.799430 | -0.066935 | 7 | 201 | 0.774038 | 1.000000 |
| strong utility action | 88 | 0.423 | 0.706843 | 0.749269 | -0.042426 | 5 | 83 | 0.579545 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 88 | 0.423 | 0.706843 | 0.749269 | -0.042426 | 5 | 83 | 0.579545 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 208 | 1.000 | 0.732495 | 0.799430 | -0.066935 | 7 | 201 | 0.774038 | 1.000000 |

## Active Smoke/Inferno Intervals

- `10.0s` - `31.5s`, rows `44`
- `56.0s` - `77.5s`, rows `44`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `60.5`, LSTM `0.8890`, XGBoost `0.9858`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `62.5`, LSTM `0.8922`, XGBoost `0.9859`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `62.0`, LSTM `0.8923`, XGBoost `0.9859`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `61.5`, LSTM `0.8935`, XGBoost `0.9859`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `59.5`, LSTM `0.8995`, XGBoost `0.9859`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `61.0`, LSTM `0.9013`, XGBoost `0.9858`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `59.0`, LSTM `0.9023`, XGBoost `0.9859`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `58.5`, LSTM `0.9023`, XGBoost `0.9859`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `64.0`, LSTM `0.9025`, XGBoost `0.9859`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `63.5`, LSTM `0.9038`, XGBoost `0.9859`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
