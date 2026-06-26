# Local Round Utility State Analysis

- csv_path: `processed_full/blast_open_london/blast-open-london-2025-vitality-vs-faze-bo3-cXlexQJNK-6GX9ddkYcv53/vitality-vs-faze-m1-mirage.csv`
- round_num: `15`
- rows: `206`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 206 | 1.000 | 0.236347 | 0.257216 | -0.020869 | 134 | 72 | 0.684466 | 0.825243 |
| active/recent utility | 206 | 1.000 | 0.236347 | 0.257216 | -0.020869 | 134 | 72 | 0.684466 | 0.825243 |
| strong utility action | 135 | 0.655 | 0.267441 | 0.294488 | -0.027048 | 82 | 53 | 0.666667 | 0.792593 |
| utility damage | 20 | 0.097 | 0.125930 | 0.165654 | -0.039723 | 15 | 5 | 1.000000 | 1.000000 |
| active smoke/inferno | 128 | 0.621 | 0.281221 | 0.306115 | -0.024894 | 75 | 53 | 0.648438 | 0.781250 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 206 | 1.000 | 0.236347 | 0.257216 | -0.020869 | 134 | 72 | 0.684466 | 0.825243 |

## Active Smoke/Inferno Intervals

- `6.0s` - `33.0s`, rows `55`
- `41.5s` - `77.5s`, rows `73`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `50.5`, LSTM `0.2671`, XGBoost `0.4749`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `49.5`, LSTM `0.0616`, XGBoost `0.2480`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `49.0`, LSTM `0.0661`, XGBoost `0.2480`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `50.0`, LSTM `0.1805`, XGBoost `0.3591`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `55.0`, LSTM `0.3210`, XGBoost `0.4958`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `48.5`, LSTM `0.0810`, XGBoost `0.2495`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `53.5`, LSTM `0.3348`, XGBoost `0.4984`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `55.5`, LSTM `0.3392`, XGBoost `0.4958`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `51.0`, LSTM `0.3195`, XGBoost `0.4749`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `54.0`, LSTM `0.3494`, XGBoost `0.4993`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
