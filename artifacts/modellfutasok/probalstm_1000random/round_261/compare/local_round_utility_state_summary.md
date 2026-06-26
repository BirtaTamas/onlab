# Local Round Utility State Analysis

- csv_path: `processed_full/blast_open_lisbon/blast-open-lisbon-2025-spirit-vs-the-huns-bo3-TWIJIxJZifB3vPv3OUvjVr/spirit-vs-the-huns-m2-dust2.csv`
- round_num: `7`
- rows: `224`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 224 | 1.000 | 0.267223 | 0.420304 | -0.153081 | 182 | 42 | 0.866071 | 0.834821 |
| active/recent utility | 224 | 1.000 | 0.267223 | 0.420304 | -0.153081 | 182 | 42 | 0.866071 | 0.834821 |
| strong utility action | 170 | 0.759 | 0.267979 | 0.455422 | -0.187443 | 149 | 21 | 0.876471 | 0.835294 |
| utility damage | 10 | 0.045 | 0.198600 | 0.338696 | -0.140095 | 10 | 0 | 1.000000 | 1.000000 |
| active smoke/inferno | 170 | 0.759 | 0.267979 | 0.455422 | -0.187443 | 149 | 21 | 0.876471 | 0.835294 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 224 | 1.000 | 0.267223 | 0.420304 | -0.153081 | 182 | 42 | 0.866071 | 0.834821 |

## Active Smoke/Inferno Intervals

- `9.0s` - `94.5s`, rows `170`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `37.0`, LSTM `0.1425`, XGBoost `0.4461`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `33.0`, LSTM `0.1479`, XGBoost `0.4507`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `36.0`, LSTM `0.1430`, XGBoost `0.4446`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `33.5`, LSTM `0.1517`, XGBoost `0.4490`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `37.5`, LSTM `0.1548`, XGBoost `0.4496`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `29.0`, LSTM `0.1828`, XGBoost `0.4581`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `9.0`, LSTM `0.2099`, XGBoost `0.4845`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `34.0`, LSTM `0.1707`, XGBoost `0.4452`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `69.0`, LSTM `0.0974`, XGBoost `0.3697`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `32.0`, recent_utility `0`
- seconds `17.0`, LSTM `0.1498`, XGBoost `0.4219`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
