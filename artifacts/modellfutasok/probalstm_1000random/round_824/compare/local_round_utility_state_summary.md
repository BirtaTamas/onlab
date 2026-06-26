# Local Round Utility State Analysis

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-natus-vincere-vs-3dmax-bo3-JB3JZO-5zNCohi5tAgyHtq/natus-vincere-vs-3dmax-m2-inferno.csv`
- round_num: `20`
- rows: `228`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 228 | 1.000 | 0.270097 | 0.289964 | -0.019867 | 207 | 21 | 0.789474 | 0.627193 |
| active/recent utility | 228 | 1.000 | 0.270097 | 0.289964 | -0.019867 | 207 | 21 | 0.789474 | 0.627193 |
| strong utility action | 184 | 0.807 | 0.313535 | 0.334876 | -0.021341 | 163 | 21 | 0.739130 | 0.538043 |
| utility damage | 23 | 0.101 | 0.456647 | 0.503270 | -0.046623 | 22 | 1 | 0.913043 | 0.304348 |
| active smoke/inferno | 174 | 0.763 | 0.307188 | 0.326542 | -0.019354 | 153 | 21 | 0.724138 | 0.511494 |
| recent utility last 5s | 10 | 0.044 | 0.423966 | 0.479887 | -0.055922 | 10 | 0 | 1.000000 | 1.000000 |
| flash effect present | 228 | 1.000 | 0.270097 | 0.289964 | -0.019867 | 207 | 21 | 0.789474 | 0.627193 |

## Active Smoke/Inferno Intervals

- `9.5s` - `96.0s`, rows `174`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `63.5`, LSTM `0.2646`, XGBoost `0.3966`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `62.5`, LSTM `0.3009`, XGBoost `0.4174`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `21.0`, LSTM `0.3635`, XGBoost `0.4750`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `60.0`, recent_utility `0`
- seconds `20.5`, LSTM `0.3685`, XGBoost `0.4750`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `60.0`, recent_utility `0`
- seconds `20.0`, LSTM `0.3735`, XGBoost `0.4764`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `60.0`, recent_utility `0`
- seconds `63.0`, LSTM `0.2995`, XGBoost `0.3958`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `64.0`, LSTM `0.2910`, XGBoost `0.3860`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `21.5`, LSTM `0.3948`, XGBoost `0.4892`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `73.0`, recent_utility `0`
- seconds `22.0`, LSTM `0.4016`, XGBoost `0.4897`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `73.0`, recent_utility `0`
- seconds `22.5`, LSTM `0.4029`, XGBoost `0.4897`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `73.0`, recent_utility `0`
