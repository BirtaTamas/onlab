# Local Round Utility State Analysis

- csv_path: `processed_full/blast_rivals_season_1/blast-rivals-2025-season-1-spirit-vs-flyquest-bo3-fQI-qOiPd1cRkmhkz0Xs5h/spirit-vs-flyquest-m1-mirage.csv`
- round_num: `3`
- rows: `112`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 112 | 1.000 | 0.637763 | 0.838063 | -0.200300 | 0 | 112 | 0.866071 | 0.946429 |
| active/recent utility | 112 | 1.000 | 0.637763 | 0.838063 | -0.200300 | 0 | 112 | 0.866071 | 0.946429 |
| strong utility action | 69 | 0.616 | 0.598528 | 0.825877 | -0.227349 | 0 | 69 | 0.855072 | 0.942029 |
| utility damage | 32 | 0.286 | 0.657896 | 0.849392 | -0.191496 | 0 | 32 | 0.875000 | 1.000000 |
| active smoke/inferno | 59 | 0.527 | 0.603926 | 0.833907 | -0.229981 | 0 | 59 | 0.898305 | 0.932203 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 112 | 1.000 | 0.637763 | 0.838063 | -0.200300 | 0 | 112 | 0.866071 | 0.946429 |

## Active Smoke/Inferno Intervals

- `7.5s` - `12.5s`, rows `11`
- `21.0s` - `44.5s`, rows `48`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `40.5`, LSTM `0.5662`, XGBoost `0.8882`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `32.0`, LSTM `0.5506`, XGBoost `0.8689`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `41.5`, LSTM `0.5673`, XGBoost `0.8813`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `33.5`, LSTM `0.5349`, XGBoost `0.8486`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `41.0`, LSTM `0.5672`, XGBoost `0.8807`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `35.5`, LSTM `0.5610`, XGBoost `0.8721`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `46.0`, recent_utility `0`
- seconds `34.5`, LSTM `0.5393`, XGBoost `0.8501`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `4.0`, recent_utility `0`
- seconds `37.0`, LSTM `0.5073`, XGBoost `0.8173`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `46.0`, recent_utility `0`
- seconds `33.0`, LSTM `0.5339`, XGBoost `0.8431`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `26.0`, LSTM `0.4949`, XGBoost `0.8030`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
