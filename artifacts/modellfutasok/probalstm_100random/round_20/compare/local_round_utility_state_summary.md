# Local Round Utility State Analysis

- csv_path: `processed_full/fissure_playground_1/fissure-playground-1-wildcard-vs-furia-bo3-u8Kr9GGu18RWnHSjYzEreW/wildcard-vs-furia-m2-inferno.csv`
- round_num: `4`
- rows: `137`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 137 | 1.000 | 0.834263 | 0.748451 | 0.085812 | 124 | 13 | 0.970803 | 0.795620 |
| active/recent utility | 137 | 1.000 | 0.834263 | 0.748451 | 0.085812 | 124 | 13 | 0.970803 | 0.795620 |
| strong utility action | 82 | 0.599 | 0.813118 | 0.726688 | 0.086430 | 77 | 5 | 0.963415 | 0.768293 |
| utility damage | 17 | 0.124 | 0.712194 | 0.656703 | 0.055491 | 15 | 2 | 0.823529 | 0.764706 |
| active smoke/inferno | 71 | 0.518 | 0.852652 | 0.765009 | 0.087642 | 66 | 5 | 0.957746 | 0.887324 |
| recent utility last 5s | 11 | 0.080 | 0.557943 | 0.479340 | 0.078603 | 11 | 0 | 1.000000 | 0.000000 |
| flash effect present | 137 | 1.000 | 0.834263 | 0.748451 | 0.085812 | 124 | 13 | 0.970803 | 0.795620 |

## Active Smoke/Inferno Intervals

- `10.0s` - `43.5s`, rows `68`
- `67.0s` - `68.0s`, rows `3`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `17.5`, LSTM `0.7008`, XGBoost `0.5595`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `66.0`, recent_utility `0`
- seconds `20.5`, LSTM `0.8992`, XGBoost `0.7627`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `38.0`, LSTM `0.9388`, XGBoost `0.8058`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `38.5`, LSTM `0.9383`, XGBoost `0.8058`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `36.5`, LSTM `0.9394`, XGBoost `0.8069`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `37.5`, LSTM `0.9360`, XGBoost `0.8058`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `35.5`, LSTM `0.9366`, XGBoost `0.8069`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `35.0`, LSTM `0.9366`, XGBoost `0.8069`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `37.0`, LSTM `0.9347`, XGBoost `0.8058`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `39.5`, LSTM `0.9338`, XGBoost `0.8058`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
