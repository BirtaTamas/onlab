# Local Round Utility State Analysis

- csv_path: `processed_full/blast_open_lisbon/blast-open-lisbon-2025-faze-vs-virtuspro-bo3-YK5CkfojMMEdQ3U0_CVA2l/faze-vs-virtuspro-m3-dust2.csv`
- round_num: `2`
- rows: `240`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 240 | 1.000 | 0.247242 | 0.279992 | -0.032749 | 230 | 10 | 0.795833 | 0.650000 |
| active/recent utility | 240 | 1.000 | 0.247242 | 0.279992 | -0.032749 | 230 | 10 | 0.795833 | 0.650000 |
| strong utility action | 118 | 0.492 | 0.233313 | 0.256916 | -0.023602 | 109 | 9 | 0.728814 | 0.686441 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 108 | 0.450 | 0.207534 | 0.233560 | -0.026027 | 106 | 2 | 0.796296 | 0.750000 |
| recent utility last 5s | 10 | 0.042 | 0.511735 | 0.509153 | 0.002582 | 3 | 7 | 0.000000 | 0.000000 |
| flash effect present | 240 | 1.000 | 0.247242 | 0.279992 | -0.032749 | 230 | 10 | 0.795833 | 0.650000 |

## Active Smoke/Inferno Intervals

- `28.5s` - `56.0s`, rows `56`
- `76.5s` - `102.0s`, rows `52`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `43.0`, LSTM `0.1901`, XGBoost `0.3043`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `43.5`, LSTM `0.2047`, XGBoost `0.3074`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `44.5`, LSTM `0.2085`, XGBoost `0.3074`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `42.0`, LSTM `0.3983`, XGBoost `0.3013`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `44.0`, LSTM `0.2200`, XGBoost `0.3074`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `77.0`, LSTM `0.3762`, XGBoost `0.4576`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `3.0`, recent_utility `0`
- seconds `56.0`, LSTM `0.2385`, XGBoost `0.3019`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `53.5`, LSTM `0.2468`, XGBoost `0.3001`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `53.0`, LSTM `0.2472`, XGBoost `0.2977`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `77.5`, LSTM `0.1383`, XGBoost `0.1850`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `6.0`, recent_utility `0`
