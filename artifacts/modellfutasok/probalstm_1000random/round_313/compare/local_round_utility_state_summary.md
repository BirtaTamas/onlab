# Local Round Utility State Analysis

- csv_path: `processed_full/blast_open_london/blast-open-london-2025-spirit-vs-furia-bo3-_zQK5XUu10iN1JLmPA8zQ4/spirit-vs-furia-m2-nuke.csv`
- round_num: `1`
- rows: `174`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 174 | 1.000 | 0.211704 | 0.266108 | -0.054404 | 171 | 3 | 0.758621 | 0.655172 |
| active/recent utility | 174 | 1.000 | 0.211704 | 0.266108 | -0.054404 | 171 | 3 | 0.758621 | 0.655172 |
| strong utility action | 56 | 0.322 | 0.417269 | 0.501158 | -0.083888 | 53 | 3 | 0.660714 | 0.339286 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 56 | 0.322 | 0.417269 | 0.501158 | -0.083888 | 53 | 3 | 0.660714 | 0.339286 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 174 | 1.000 | 0.211704 | 0.266108 | -0.054404 | 171 | 3 | 0.758621 | 0.655172 |

## Active Smoke/Inferno Intervals

- `11.5s` - `39.0s`, rows `56`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `15.5`, LSTM `0.2236`, XGBoost `0.5206`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `76.0`, recent_utility `0`
- seconds `14.5`, LSTM `0.2307`, XGBoost `0.5222`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `60.0`, recent_utility `0`
- seconds `16.0`, LSTM `0.1694`, XGBoost `0.4560`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `76.0`, recent_utility `0`
- seconds `15.0`, LSTM `0.1930`, XGBoost `0.4683`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `76.0`, recent_utility `0`
- seconds `14.0`, LSTM `0.3002`, XGBoost `0.5238`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `36.0`, recent_utility `0`
- seconds `16.5`, LSTM `0.5389`, XGBoost `0.7351`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `76.0`, recent_utility `0`
- seconds `13.5`, LSTM `0.0855`, XGBoost `0.2803`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `22.0`, recent_utility `0`
- seconds `13.0`, LSTM `0.0880`, XGBoost `0.2752`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `7.0`, recent_utility `0`
- seconds `17.0`, LSTM `0.5536`, XGBoost `0.7351`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `76.0`, recent_utility `0`
- seconds `17.5`, LSTM `0.4823`, XGBoost `0.6392`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `75.0`, recent_utility `0`
