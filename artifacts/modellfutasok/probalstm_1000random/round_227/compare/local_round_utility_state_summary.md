# Local Round Utility State Analysis

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-tyloo-vs-vitality-bo3-aF98ikh3PjdqKlkdIJn9tC/tyloo-vs-vitality-m1-inferno.csv`
- round_num: `3`
- rows: `151`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 151 | 1.000 | 0.952657 | 0.968722 | -0.016064 | 34 | 117 | 1.000000 | 1.000000 |
| active/recent utility | 151 | 1.000 | 0.952657 | 0.968722 | -0.016064 | 34 | 117 | 1.000000 | 1.000000 |
| strong utility action | 122 | 0.808 | 0.959655 | 0.966893 | -0.007238 | 34 | 88 | 1.000000 | 1.000000 |
| utility damage | 23 | 0.152 | 0.968321 | 0.981122 | -0.012802 | 1 | 22 | 1.000000 | 1.000000 |
| active smoke/inferno | 122 | 0.808 | 0.959655 | 0.966893 | -0.007238 | 34 | 88 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 151 | 1.000 | 0.952657 | 0.968722 | -0.016064 | 34 | 117 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `10.5s` - `71.0s`, rows `122`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `15.5`, LSTM `0.9298`, XGBoost `0.9771`, closer `xgboost`, smoke `0`, inferno `3`, utility_damage `0.0`, recent_utility `0`
- seconds `13.0`, LSTM `0.9266`, XGBoost `0.9736`, closer `xgboost`, smoke `0`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `12.5`, LSTM `0.9292`, XGBoost `0.9736`, closer `xgboost`, smoke `0`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `16.5`, LSTM `0.9341`, XGBoost `0.9771`, closer `xgboost`, smoke `0`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `10.5`, LSTM `0.9304`, XGBoost `0.9733`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `12.0`, LSTM `0.9308`, XGBoost `0.9735`, closer `xgboost`, smoke `0`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `14.0`, LSTM `0.9353`, XGBoost `0.9770`, closer `xgboost`, smoke `0`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `11.5`, LSTM `0.9323`, XGBoost `0.9737`, closer `xgboost`, smoke `0`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `16.0`, LSTM `0.9360`, XGBoost `0.9771`, closer `xgboost`, smoke `0`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `13.5`, LSTM `0.9366`, XGBoost `0.9770`, closer `xgboost`, smoke `0`, inferno `2`, utility_damage `0.0`, recent_utility `0`
