# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-furia-vs-aurora-bo3-0icw3xvkvOZhHsCT2PEavZ/furia-vs-aurora-m1-inferno.csv`
- round_num: `16`
- rows: `117`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 117 | 1.000 | 0.125130 | 0.181355 | -0.056225 | 116 | 1 | 1.000000 | 1.000000 |
| active/recent utility | 117 | 1.000 | 0.125130 | 0.181355 | -0.056225 | 116 | 1 | 1.000000 | 1.000000 |
| strong utility action | 98 | 0.838 | 0.096192 | 0.147610 | -0.051417 | 97 | 1 | 1.000000 | 1.000000 |
| utility damage | 10 | 0.085 | 0.332177 | 0.390796 | -0.058619 | 10 | 0 | 1.000000 | 1.000000 |
| active smoke/inferno | 98 | 0.838 | 0.096192 | 0.147610 | -0.051417 | 97 | 1 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 117 | 1.000 | 0.125130 | 0.181355 | -0.056225 | 116 | 1 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `9.5s` - `58.0s`, rows `98`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `13.0`, LSTM `0.2188`, XGBoost `0.3525`, closer `lstm`, smoke `1`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `10.0`, LSTM `0.2307`, XGBoost `0.3612`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `11.5`, LSTM `0.2320`, XGBoost `0.3615`, closer `lstm`, smoke `1`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `10.5`, LSTM `0.2343`, XGBoost `0.3618`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `9.5`, LSTM `0.2356`, XGBoost `0.3609`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `24.0`, LSTM `0.0878`, XGBoost `0.2098`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `11.0`, LSTM `0.2418`, XGBoost `0.3615`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `23.5`, LSTM `0.0919`, XGBoost `0.2098`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `14.5`, LSTM `0.2373`, XGBoost `0.3532`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `25.5`, LSTM `0.0940`, XGBoost `0.2098`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
