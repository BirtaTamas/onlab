# Local Round Utility State Analysis

- csv_path: `processed_full/blast_open_london/blast-open-london-2025-spirit-vs-furia-bo3-_zQK5XUu10iN1JLmPA8zQ4/spirit-vs-furia-m2-nuke.csv`
- round_num: `22`
- rows: `212`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 212 | 1.000 | 0.501757 | 0.447165 | 0.054592 | 110 | 102 | 0.528302 | 0.537736 |
| active/recent utility | 212 | 1.000 | 0.501757 | 0.447165 | 0.054592 | 110 | 102 | 0.528302 | 0.537736 |
| strong utility action | 191 | 0.901 | 0.487025 | 0.427621 | 0.059405 | 105 | 86 | 0.476440 | 0.486911 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 191 | 0.901 | 0.487025 | 0.427621 | 0.059405 | 105 | 86 | 0.476440 | 0.486911 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 212 | 1.000 | 0.501757 | 0.447165 | 0.054592 | 110 | 102 | 0.528302 | 0.537736 |

## Active Smoke/Inferno Intervals

- `7.5s` - `102.5s`, rows `191`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `70.5`, LSTM `0.4970`, XGBoost `0.3057`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `71.0`, LSTM `0.4951`, XGBoost `0.3076`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `70.0`, LSTM `0.4955`, XGBoost `0.3080`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `69.0`, LSTM `0.4961`, XGBoost `0.3090`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `76.0`, LSTM `0.4928`, XGBoost `0.3060`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `95.0`, LSTM `0.4902`, XGBoost `0.3036`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `69.5`, LSTM `0.4952`, XGBoost `0.3090`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `94.0`, LSTM `0.4990`, XGBoost `0.3136`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `72.5`, LSTM `0.4930`, XGBoost `0.3078`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `49.5`, LSTM `0.4932`, XGBoost `0.3087`, closer `lstm`, smoke `4`, inferno `1`, utility_damage `0.0`, recent_utility `0`
