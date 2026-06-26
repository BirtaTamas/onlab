# Local Round Utility State Analysis

- csv_path: `processed_full/iem_katowice/iem-katowice-2025-gamerlegion-vs-the-mongolz-bo3-zdjI5BKx0DIgDYoNAnfKpI/gamerlegion-vs-the-mongolz-m2-mirage.csv`
- round_num: `13`
- rows: `132`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 132 | 1.000 | 0.419161 | 0.571319 | -0.152158 | 30 | 102 | 0.545455 | 0.765152 |
| active/recent utility | 132 | 1.000 | 0.419161 | 0.571319 | -0.152158 | 30 | 102 | 0.545455 | 0.765152 |
| strong utility action | 76 | 0.576 | 0.335294 | 0.564815 | -0.229521 | 2 | 74 | 0.276316 | 0.750000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 76 | 0.576 | 0.335294 | 0.564815 | -0.229521 | 2 | 74 | 0.276316 | 0.750000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 132 | 1.000 | 0.419161 | 0.571319 | -0.152158 | 30 | 102 | 0.545455 | 0.765152 |

## Active Smoke/Inferno Intervals

- `20.5s` - `58.0s`, rows `76`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `48.0`, LSTM `0.1413`, XGBoost `0.5296`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `48.5`, LSTM `0.1423`, XGBoost `0.5296`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `47.5`, LSTM `0.1468`, XGBoost `0.5296`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `47.0`, LSTM `0.1524`, XGBoost `0.5296`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `49.0`, LSTM `0.1631`, XGBoost `0.5279`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `45.5`, LSTM `0.1624`, XGBoost `0.5269`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `45.0`, LSTM `0.1663`, XGBoost `0.5269`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `52.0`, LSTM `0.1765`, XGBoost `0.5366`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `46.5`, LSTM `0.1693`, XGBoost `0.5269`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `51.5`, LSTM `0.1724`, XGBoost `0.5285`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
