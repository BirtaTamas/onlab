# Local Round Utility State Analysis

- csv_path: `processed_full/blast_rivals_season_1/blast-rivals-2025-season-1-mouz-vs-pain-bo3-Ao8EIC0rxvFkpkJ5bGImFu/mouz-vs-pain-m3-dust2.csv`
- round_num: `13`
- rows: `162`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 162 | 1.000 | 0.138369 | 0.223791 | -0.085423 | 162 | 0 | 1.000000 | 0.981481 |
| active/recent utility | 162 | 1.000 | 0.138369 | 0.223791 | -0.085423 | 162 | 0 | 1.000000 | 0.981481 |
| strong utility action | 144 | 0.889 | 0.135599 | 0.221933 | -0.086334 | 144 | 0 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 138 | 0.852 | 0.128672 | 0.211758 | -0.083086 | 138 | 0 | 1.000000 | 1.000000 |
| recent utility last 5s | 10 | 0.062 | 0.220810 | 0.384675 | -0.163865 | 10 | 0 | 1.000000 | 1.000000 |
| flash effect present | 162 | 1.000 | 0.138369 | 0.223791 | -0.085423 | 162 | 0 | 1.000000 | 0.981481 |

## Active Smoke/Inferno Intervals

- `5.5s` - `32.0s`, rows `54`
- `35.5s` - `77.0s`, rows `84`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `4.5`, LSTM `0.2771`, XGBoost `0.4908`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `7.5`, LSTM `0.0852`, XGBoost `0.2734`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `54.0`, LSTM `0.1373`, XGBoost `0.3206`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `7.0`, LSTM `0.0926`, XGBoost `0.2734`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `4.0`, LSTM `0.3104`, XGBoost `0.4908`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `8.0`, LSTM `0.0899`, XGBoost `0.2681`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `9.0`, LSTM `0.0921`, XGBoost `0.2700`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `8.5`, LSTM `0.0986`, XGBoost `0.2729`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `9.5`, LSTM `0.1008`, XGBoost `0.2732`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `6.0`, LSTM `0.1087`, XGBoost `0.2792`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `1`
