# Local Round Utility State Analysis

- csv_path: `processed_full/blast_open_lisbon/blast-open-lisbon-2025-spirit-vs-the-huns-bo3-TWIJIxJZifB3vPv3OUvjVr/spirit-vs-the-huns-m2-dust2.csv`
- round_num: `2`
- rows: `227`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 227 | 1.000 | 0.605563 | 0.718578 | -0.113015 | 226 | 1 | 0.352423 | 0.273128 |
| active/recent utility | 227 | 1.000 | 0.605563 | 0.718578 | -0.113015 | 226 | 1 | 0.352423 | 0.273128 |
| strong utility action | 92 | 0.405 | 0.495322 | 0.627200 | -0.131878 | 92 | 0 | 0.489130 | 0.434783 |
| utility damage | 20 | 0.088 | 0.471071 | 0.545828 | -0.074757 | 20 | 0 | 0.500000 | 0.500000 |
| active smoke/inferno | 92 | 0.405 | 0.495322 | 0.627200 | -0.131878 | 92 | 0 | 0.489130 | 0.434783 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 227 | 1.000 | 0.605563 | 0.718578 | -0.113015 | 226 | 1 | 0.352423 | 0.273128 |

## Active Smoke/Inferno Intervals

- `10.0s` - `33.0s`, rows `47`
- `81.0s` - `103.0s`, rows `45`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `82.5`, LSTM `0.1277`, XGBoost `0.5488`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `83.0`, LSTM `0.1335`, XGBoost `0.5505`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `82.0`, LSTM `0.1331`, XGBoost `0.5488`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `81.5`, LSTM `0.1325`, XGBoost `0.5408`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `81.0`, LSTM `0.1298`, XGBoost `0.5083`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `83.5`, LSTM `0.1077`, XGBoost `0.4404`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `84.0`, LSTM `0.1124`, XGBoost `0.4443`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `84.5`, LSTM `0.1200`, XGBoost `0.4434`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `86.0`, LSTM `0.1378`, XGBoost `0.4434`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `86.5`, LSTM `0.1386`, XGBoost `0.4426`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
