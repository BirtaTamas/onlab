# Local Round Utility State Analysis

- csv_path: `processed_full/iem_cologne/iem-cologne-2025-ninjas-in-pyjamas-vs-natus-vincere-bo3-NyGFAYn592Hu0YfljTVG0N/ninjas-in-pyjamas-vs-natus-vincere-m2-inferno.csv`
- round_num: `1`
- rows: `207`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 207 | 1.000 | 0.273451 | 0.349444 | -0.075992 | 194 | 13 | 0.946860 | 0.579710 |
| active/recent utility | 207 | 1.000 | 0.273451 | 0.349444 | -0.075992 | 194 | 13 | 0.946860 | 0.579710 |
| strong utility action | 53 | 0.256 | 0.185800 | 0.249663 | -0.063863 | 53 | 0 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 53 | 0.256 | 0.185800 | 0.249663 | -0.063863 | 53 | 0 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 207 | 1.000 | 0.273451 | 0.349444 | -0.075992 | 194 | 13 | 0.946860 | 0.579710 |

## Active Smoke/Inferno Intervals

- `67.0s` - `93.0s`, rows `53`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `70.5`, LSTM `0.2722`, XGBoost `0.4639`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `70.0`, LSTM `0.2780`, XGBoost `0.4628`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `71.0`, LSTM `0.2982`, XGBoost `0.4639`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `79.0`, LSTM `0.2138`, XGBoost `0.3507`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `71.5`, LSTM `0.3311`, XGBoost `0.4639`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `74.0`, LSTM `0.2945`, XGBoost `0.4241`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `72.0`, LSTM `0.3395`, XGBoost `0.4639`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `73.5`, LSTM `0.3167`, XGBoost `0.4332`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `73.0`, LSTM `0.3332`, XGBoost `0.4360`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `78.0`, LSTM `0.2584`, XGBoost `0.3577`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
