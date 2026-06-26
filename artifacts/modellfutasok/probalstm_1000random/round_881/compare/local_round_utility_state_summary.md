# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-spirit-vs-faze-bo3-1414ljxN3FRmXv6-03KYFL/spirit-vs-faze-m2-mirage.csv`
- round_num: `18`
- rows: `238`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 238 | 1.000 | 0.111393 | 0.186211 | -0.074818 | 238 | 0 | 1.000000 | 1.000000 |
| active/recent utility | 238 | 1.000 | 0.111393 | 0.186211 | -0.074818 | 238 | 0 | 1.000000 | 1.000000 |
| strong utility action | 209 | 0.878 | 0.104118 | 0.179763 | -0.075645 | 209 | 0 | 1.000000 | 1.000000 |
| utility damage | 10 | 0.042 | 0.043130 | 0.103371 | -0.060241 | 10 | 0 | 1.000000 | 1.000000 |
| active smoke/inferno | 209 | 0.878 | 0.104118 | 0.179763 | -0.075645 | 209 | 0 | 1.000000 | 1.000000 |
| recent utility last 5s | 10 | 0.042 | 0.331010 | 0.479674 | -0.148664 | 10 | 0 | 1.000000 | 1.000000 |
| flash effect present | 238 | 1.000 | 0.111393 | 0.186211 | -0.074818 | 238 | 0 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `5.5s` - `71.0s`, rows `132`
- `73.0s` - `79.5s`, rows `14`
- `81.5s` - `112.5s`, rows `63`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `81.5`, LSTM `0.1035`, XGBoost `0.4242`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `83.5`, LSTM `0.1230`, XGBoost `0.4363`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `84.0`, LSTM `0.1295`, XGBoost `0.4304`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `84.5`, LSTM `0.1446`, XGBoost `0.4363`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `82.0`, LSTM `0.1800`, XGBoost `0.4242`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `82.5`, LSTM `0.1892`, XGBoost `0.4242`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `83.0`, LSTM `0.1952`, XGBoost `0.4242`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `11.0`, LSTM `0.2870`, XGBoost `0.4814`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `1`
- seconds `75.5`, LSTM `0.0315`, XGBoost `0.2217`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `13.0`, recent_utility `0`
- seconds `85.0`, LSTM `0.1305`, XGBoost `0.3206`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
