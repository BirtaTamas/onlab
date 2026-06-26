# Local Round Utility State Analysis

- csv_path: `processed_full/blast_open_london/blast-open-london-2025-spirit-vs-furia-bo3-_zQK5XUu10iN1JLmPA8zQ4/spirit-vs-furia-m2-nuke.csv`
- round_num: `19`
- rows: `181`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 181 | 1.000 | 0.281257 | 0.301163 | -0.019905 | 123 | 58 | 0.850829 | 0.701657 |
| active/recent utility | 181 | 1.000 | 0.281257 | 0.301163 | -0.019905 | 123 | 58 | 0.850829 | 0.701657 |
| strong utility action | 128 | 0.707 | 0.353730 | 0.369862 | -0.016132 | 85 | 43 | 0.804688 | 0.625000 |
| utility damage | 10 | 0.055 | 0.388740 | 0.454265 | -0.065525 | 10 | 0 | 0.900000 | 0.400000 |
| active smoke/inferno | 118 | 0.652 | 0.338767 | 0.353215 | -0.014448 | 75 | 43 | 0.872881 | 0.677966 |
| recent utility last 5s | 10 | 0.055 | 0.530289 | 0.566288 | -0.035999 | 10 | 0 | 0.000000 | 0.000000 |
| flash effect present | 181 | 1.000 | 0.281257 | 0.301163 | -0.019905 | 123 | 58 | 0.850829 | 0.701657 |

## Active Smoke/Inferno Intervals

- `8.0s` - `66.5s`, rows `118`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `49.5`, LSTM `0.4524`, XGBoost `0.2302`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `50.0`, LSTM `0.4150`, XGBoost `0.2302`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `51.0`, LSTM `0.4149`, XGBoost `0.2302`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `50.5`, LSTM `0.4115`, XGBoost `0.2302`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `51.5`, LSTM `0.4079`, XGBoost `0.2286`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `44.0`, LSTM `0.2639`, XGBoost `0.4380`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `34.5`, LSTM `0.0908`, XGBoost `0.2629`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `52.0`, LSTM `0.4069`, XGBoost `0.2358`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `32.0`, LSTM `0.1301`, XGBoost `0.3012`, closer `lstm`, smoke `4`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `31.0`, LSTM `0.1347`, XGBoost `0.3050`, closer `lstm`, smoke `4`, inferno `1`, utility_damage `0.0`, recent_utility `0`
