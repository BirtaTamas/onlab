# Local Round Utility State Analysis

- csv_path: `processed_full/blast_open_lisbon/blast-open-lisbon-2025-mouz-vs-falcons-bo3-xBECUqZMcQ8GCwi-GUyz8e/mouz-vs-falcons-m1-train.csv`
- round_num: `3`
- rows: `275`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 275 | 1.000 | 0.195700 | 0.178954 | 0.016746 | 137 | 138 | 0.923636 | 0.963636 |
| active/recent utility | 275 | 1.000 | 0.195700 | 0.178954 | 0.016746 | 137 | 138 | 0.923636 | 0.963636 |
| strong utility action | 204 | 0.742 | 0.239863 | 0.218630 | 0.021233 | 77 | 127 | 0.926471 | 0.975490 |
| utility damage | 12 | 0.044 | 0.316279 | 0.278656 | 0.037622 | 2 | 10 | 1.000000 | 1.000000 |
| active smoke/inferno | 194 | 0.705 | 0.225367 | 0.204492 | 0.020876 | 77 | 117 | 0.974227 | 0.974227 |
| recent utility last 5s | 10 | 0.036 | 0.521075 | 0.492908 | 0.028167 | 0 | 10 | 0.000000 | 1.000000 |
| flash effect present | 275 | 1.000 | 0.195700 | 0.178954 | 0.016746 | 137 | 138 | 0.923636 | 0.963636 |

## Active Smoke/Inferno Intervals

- `8.0s` - `36.0s`, rows `57`
- `39.0s` - `107.0s`, rows `137`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `50.5`, LSTM `0.3837`, XGBoost `0.2573`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `50.0`, LSTM `0.3807`, XGBoost `0.2573`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `51.0`, LSTM `0.3803`, XGBoost `0.2573`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `49.0`, LSTM `0.3678`, XGBoost `0.2547`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `51.5`, LSTM `0.3714`, XGBoost `0.2597`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `52.0`, LSTM `0.3671`, XGBoost `0.2612`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `16.0`, LSTM `0.3846`, XGBoost `0.2813`, closer `xgboost`, smoke `3`, inferno `2`, utility_damage `21.0`, recent_utility `0`
- seconds `48.5`, LSTM `0.3438`, XGBoost `0.2497`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `10.5`, LSTM `0.3751`, XGBoost `0.2839`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `52.5`, LSTM `0.3513`, XGBoost `0.2617`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
