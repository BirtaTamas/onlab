# Local Round Utility State Analysis

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-falcons-vs-legacy-bo3-ryWGopRV1OfbL288nR6Rql/falcons-vs-legacy-m1-inferno.csv`
- round_num: `10`
- rows: `156`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 156 | 1.000 | 0.714787 | 0.761740 | -0.046952 | 23 | 133 | 1.000000 | 0.891026 |
| active/recent utility | 156 | 1.000 | 0.714787 | 0.761740 | -0.046952 | 23 | 133 | 1.000000 | 0.891026 |
| strong utility action | 136 | 0.872 | 0.740197 | 0.800575 | -0.060378 | 3 | 133 | 1.000000 | 1.000000 |
| utility damage | 10 | 0.064 | 0.683458 | 0.737168 | -0.053710 | 0 | 10 | 1.000000 | 1.000000 |
| active smoke/inferno | 136 | 0.872 | 0.740197 | 0.800575 | -0.060378 | 3 | 133 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 156 | 1.000 | 0.714787 | 0.761740 | -0.046952 | 23 | 133 | 1.000000 | 0.891026 |

## Active Smoke/Inferno Intervals

- `10.0s` - `77.5s`, rows `136`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `59.5`, LSTM `0.6643`, XGBoost `0.8135`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `61.5`, LSTM `0.6721`, XGBoost `0.8167`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `62.0`, LSTM `0.6737`, XGBoost `0.8167`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `50.0`, LSTM `0.5809`, XGBoost `0.7230`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `61.0`, LSTM `0.6754`, XGBoost `0.8152`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `50.5`, LSTM `0.5914`, XGBoost `0.7262`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `49.0`, LSTM `0.5886`, XGBoost `0.7230`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `62.5`, LSTM `0.6856`, XGBoost `0.8167`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `63.5`, LSTM `0.6885`, XGBoost `0.8193`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `60.5`, LSTM `0.6846`, XGBoost `0.8152`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
