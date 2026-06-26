# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-gamerlegion-vs-tyloo-bo3-0g9mXt3FIxC8XzjXNUjRL7/gamerlegion-vs-tyloo-m1-ancient-p3.csv`
- round_num: `10`
- rows: `255`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 255 | 1.000 | 0.558446 | 0.468598 | 0.089848 | 232 | 23 | 0.650980 | 0.639216 |
| active/recent utility | 255 | 1.000 | 0.558446 | 0.468598 | 0.089848 | 232 | 23 | 0.650980 | 0.639216 |
| strong utility action | 216 | 0.847 | 0.544991 | 0.453639 | 0.091352 | 197 | 19 | 0.597222 | 0.597222 |
| utility damage | 10 | 0.039 | 0.589546 | 0.534647 | 0.054899 | 10 | 0 | 1.000000 | 1.000000 |
| active smoke/inferno | 213 | 0.835 | 0.547284 | 0.457174 | 0.090110 | 194 | 19 | 0.605634 | 0.605634 |
| recent utility last 5s | 20 | 0.078 | 0.383916 | 0.198093 | 0.185824 | 20 | 0 | 0.000000 | 0.000000 |
| flash effect present | 255 | 1.000 | 0.558446 | 0.468598 | 0.089848 | 232 | 23 | 0.650980 | 0.639216 |

## Active Smoke/Inferno Intervals

- `8.0s` - `32.5s`, rows `50`
- `40.0s` - `99.0s`, rows `119`
- `103.5s` - `125.0s`, rows `44`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `100.5`, LSTM `0.4532`, XGBoost `0.2055`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `115.5`, LSTM `0.6373`, XGBoost `0.3921`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `94.0`, LSTM `0.4437`, XGBoost `0.1996`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `95.0`, LSTM `0.4420`, XGBoost `0.1996`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `94.5`, LSTM `0.4416`, XGBoost `0.1996`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `93.0`, LSTM `0.4375`, XGBoost `0.1957`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `87.5`, LSTM `0.4279`, XGBoost `0.1957`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `95.5`, LSTM `0.4262`, XGBoost `0.1998`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `84.0`, LSTM `0.4159`, XGBoost `0.1931`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `80.5`, LSTM `0.4244`, XGBoost `0.2019`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
