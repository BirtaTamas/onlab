# Local Round Utility State Analysis

- csv_path: `processed_full/iem_cologne/iem-cologne-2025-ninjas-in-pyjamas-vs-natus-vincere-bo3-NyGFAYn592Hu0YfljTVG0N/ninjas-in-pyjamas-vs-natus-vincere-m2-inferno.csv`
- round_num: `13`
- rows: `180`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 180 | 1.000 | 0.221700 | 0.236490 | -0.014790 | 122 | 58 | 0.838889 | 1.000000 |
| active/recent utility | 180 | 1.000 | 0.221700 | 0.236490 | -0.014790 | 122 | 58 | 0.838889 | 1.000000 |
| strong utility action | 45 | 0.250 | 0.014858 | 0.041105 | -0.026247 | 45 | 0 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 45 | 0.250 | 0.014858 | 0.041105 | -0.026247 | 45 | 0 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 180 | 1.000 | 0.221700 | 0.236490 | -0.014790 | 122 | 58 | 0.838889 | 1.000000 |

## Active Smoke/Inferno Intervals

- `60.0s` - `82.0s`, rows `45`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `69.0`, LSTM `0.0121`, XGBoost `0.0521`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `68.5`, LSTM `0.0101`, XGBoost `0.0499`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `79.0`, LSTM `0.0149`, XGBoost `0.0533`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `69.5`, LSTM `0.0144`, XGBoost `0.0521`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `78.5`, LSTM `0.0155`, XGBoost `0.0528`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `70.0`, LSTM `0.0157`, XGBoost `0.0528`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `71.5`, LSTM `0.0179`, XGBoost `0.0548`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `72.0`, LSTM `0.0182`, XGBoost `0.0548`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `73.5`, LSTM `0.0175`, XGBoost `0.0537`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `78.0`, LSTM `0.0164`, XGBoost `0.0522`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
