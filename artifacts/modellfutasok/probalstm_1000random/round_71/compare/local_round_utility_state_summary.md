# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-gamerlegion-vs-tyloo-bo3-0g9mXt3FIxC8XzjXNUjRL7/gamerlegion-vs-tyloo-m1-ancient-p3.csv`
- round_num: `8`
- rows: `180`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 180 | 1.000 | 0.524732 | 0.459441 | 0.065291 | 155 | 25 | 0.627778 | 0.455556 |
| active/recent utility | 180 | 1.000 | 0.524732 | 0.459441 | 0.065291 | 155 | 25 | 0.627778 | 0.455556 |
| strong utility action | 142 | 0.789 | 0.461986 | 0.376891 | 0.085095 | 139 | 3 | 0.528169 | 0.330986 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 142 | 0.789 | 0.461986 | 0.376891 | 0.085095 | 139 | 3 | 0.528169 | 0.330986 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 180 | 1.000 | 0.524732 | 0.459441 | 0.065291 | 155 | 25 | 0.627778 | 0.455556 |

## Active Smoke/Inferno Intervals

- `6.5s` - `77.0s`, rows `142`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `47.0`, LSTM `0.4666`, XGBoost `0.2076`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `17.0`, recent_utility `0`
- seconds `36.0`, LSTM `0.4368`, XGBoost `0.1861`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `32.0`, recent_utility `0`
- seconds `46.5`, LSTM `0.4597`, XGBoost `0.2101`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `11.0`, recent_utility `0`
- seconds `37.5`, LSTM `0.4278`, XGBoost `0.1846`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `32.0`, recent_utility `0`
- seconds `36.5`, LSTM `0.4223`, XGBoost `0.1861`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `32.0`, recent_utility `0`
- seconds `37.0`, LSTM `0.4196`, XGBoost `0.1868`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `32.0`, recent_utility `0`
- seconds `35.5`, LSTM `0.4146`, XGBoost `0.1864`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `32.0`, recent_utility `0`
- seconds `46.0`, LSTM `0.4391`, XGBoost `0.2124`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `11.0`, recent_utility `0`
- seconds `63.0`, LSTM `0.4394`, XGBoost `0.2161`, closer `lstm`, smoke `1`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `47.5`, LSTM `0.4138`, XGBoost `0.1925`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `32.0`, recent_utility `0`
