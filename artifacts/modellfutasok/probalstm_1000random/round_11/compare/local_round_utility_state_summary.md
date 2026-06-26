# Local Round Utility State Analysis

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-vitality-vs-astralis-bo3-Zley6FZuKcttfrliAqsvWJ/astralis-vs-vitality-m1-inferno.csv`
- round_num: `9`
- rows: `105`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 105 | 1.000 | 0.575481 | 0.618666 | -0.043184 | 21 | 84 | 0.485714 | 0.800000 |
| active/recent utility | 105 | 1.000 | 0.575481 | 0.618666 | -0.043184 | 21 | 84 | 0.485714 | 0.800000 |
| strong utility action | 84 | 0.800 | 0.599254 | 0.626241 | -0.026987 | 19 | 65 | 0.583333 | 0.750000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 75 | 0.714 | 0.565437 | 0.591750 | -0.026312 | 18 | 57 | 0.533333 | 0.720000 |
| recent utility last 5s | 10 | 0.095 | 0.867925 | 0.902667 | -0.034742 | 1 | 9 | 1.000000 | 1.000000 |
| flash effect present | 105 | 1.000 | 0.575481 | 0.618666 | -0.043184 | 21 | 84 | 0.485714 | 0.800000 |

## Active Smoke/Inferno Intervals

- `9.5s` - `46.5s`, rows `75`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `28.5`, LSTM `0.5405`, XGBoost `0.3906`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `27.0`, LSTM `0.5145`, XGBoost `0.3858`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `26.5`, LSTM `0.5038`, XGBoost `0.3804`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `28.0`, LSTM `0.5150`, XGBoost `0.3937`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `9.5`, LSTM `0.4297`, XGBoost `0.5471`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `27.5`, LSTM `0.5035`, XGBoost `0.3866`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `39.0`, LSTM `0.7953`, XGBoost `0.9067`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `13.0`, recent_utility `0`
- seconds `38.5`, LSTM `0.7955`, XGBoost `0.9067`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `13.0`, recent_utility `0`
- seconds `38.0`, LSTM `0.7989`, XGBoost `0.9056`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `6.0`, recent_utility `0`
- seconds `46.0`, LSTM `0.7069`, XGBoost `0.8093`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
