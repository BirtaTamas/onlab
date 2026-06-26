# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major_stage_2/blasttv-austin-major-2025-stage-2-faze-vs-heroic-dust2-PtQF8ASKD1754yZQHk6148/faze-vs-heroic-dust2.csv`
- round_num: `11`
- rows: `135`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 135 | 1.000 | 0.797101 | 0.816104 | -0.019003 | 13 | 122 | 0.962963 | 1.000000 |
| active/recent utility | 135 | 1.000 | 0.797101 | 0.816104 | -0.019003 | 13 | 122 | 0.962963 | 1.000000 |
| strong utility action | 118 | 0.874 | 0.799537 | 0.820029 | -0.020492 | 10 | 108 | 0.957627 | 1.000000 |
| utility damage | 10 | 0.074 | 0.515666 | 0.546709 | -0.031042 | 0 | 10 | 1.000000 | 1.000000 |
| active smoke/inferno | 113 | 0.837 | 0.812148 | 0.832108 | -0.019960 | 10 | 103 | 0.955752 | 1.000000 |
| recent utility last 5s | 17 | 0.126 | 0.835662 | 0.868772 | -0.033110 | 0 | 17 | 1.000000 | 1.000000 |
| flash effect present | 135 | 1.000 | 0.797101 | 0.816104 | -0.019003 | 13 | 122 | 0.962963 | 1.000000 |

## Active Smoke/Inferno Intervals

- `6.5s` - `62.5s`, rows `113`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `21.5`, LSTM `0.7790`, XGBoost `0.6359`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `22.0`, LSTM `0.7801`, XGBoost `0.6605`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `22.5`, LSTM `0.7944`, XGBoost `0.6813`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `23.0`, LSTM `0.7685`, XGBoost `0.6799`, closer `lstm`, smoke `4`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `20.5`, LSTM `0.5961`, XGBoost `0.5232`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `11.5`, LSTM `0.4829`, XGBoost `0.5521`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `11.0`, LSTM `0.4871`, XGBoost `0.5521`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `12.0`, LSTM `0.4885`, XGBoost `0.5521`, closer `xgboost`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `35.5`, LSTM `0.9158`, XGBoost `0.9763`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `12.5`, LSTM `0.4949`, XGBoost `0.5521`, closer `xgboost`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
