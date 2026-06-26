# Local Round Utility State Analysis

- csv_path: `processed_full/iem_dallas/iem-dallas-2025-liquid-vs-mouz-bo3-heKnTsZGq8rrQ4y9Yn2KrI/liquid-vs-mouz-m2-train.csv`
- round_num: `15`
- rows: `130`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 130 | 1.000 | 0.807233 | 0.765917 | 0.041316 | 73 | 57 | 0.992308 | 0.976923 |
| active/recent utility | 130 | 1.000 | 0.807233 | 0.765917 | 0.041316 | 73 | 57 | 0.992308 | 0.976923 |
| strong utility action | 98 | 0.754 | 0.801081 | 0.748642 | 0.052440 | 59 | 39 | 0.989796 | 0.969388 |
| utility damage | 18 | 0.138 | 0.609601 | 0.539754 | 0.069847 | 18 | 0 | 0.944444 | 0.833333 |
| active smoke/inferno | 98 | 0.754 | 0.801081 | 0.748642 | 0.052440 | 59 | 39 | 0.989796 | 0.969388 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 130 | 1.000 | 0.807233 | 0.765917 | 0.041316 | 73 | 57 | 0.992308 | 0.976923 |

## Active Smoke/Inferno Intervals

- `6.0s` - `36.0s`, rows `61`
- `46.5s` - `64.5s`, rows `37`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `25.5`, LSTM `0.7552`, XGBoost `0.5161`, closer `lstm`, smoke `6`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `29.5`, LSTM `0.7426`, XGBoost `0.5201`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `30.0`, LSTM `0.7390`, XGBoost `0.5201`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `26.0`, LSTM `0.7322`, XGBoost `0.5164`, closer `lstm`, smoke `6`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `29.0`, LSTM `0.7370`, XGBoost `0.5217`, closer `lstm`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `28.5`, LSTM `0.7364`, XGBoost `0.5217`, closer `lstm`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `26.5`, LSTM `0.7304`, XGBoost `0.5201`, closer `lstm`, smoke `6`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `27.0`, LSTM `0.7216`, XGBoost `0.5201`, closer `lstm`, smoke `6`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `28.0`, LSTM `0.7134`, XGBoost `0.5215`, closer `lstm`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `27.5`, LSTM `0.7069`, XGBoost `0.5213`, closer `lstm`, smoke `6`, inferno `0`, utility_damage `0.0`, recent_utility `0`
