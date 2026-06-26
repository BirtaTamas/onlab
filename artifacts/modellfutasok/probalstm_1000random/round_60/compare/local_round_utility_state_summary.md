# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-natus-vincere-vs-tyloo-bo3-u9zlDGjnIy0eSohnO5P-Xx/natus-vincere-vs-tyloo-m3-ancient.csv`
- round_num: `18`
- rows: `245`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 245 | 1.000 | 0.326257 | 0.450278 | -0.124021 | 209 | 36 | 0.914286 | 0.587755 |
| active/recent utility | 245 | 1.000 | 0.326257 | 0.450278 | -0.124021 | 209 | 36 | 0.914286 | 0.587755 |
| strong utility action | 169 | 0.690 | 0.341882 | 0.455608 | -0.113726 | 137 | 32 | 0.881657 | 0.674556 |
| utility damage | 17 | 0.069 | 0.632858 | 0.601429 | 0.031428 | 5 | 12 | 0.294118 | 0.294118 |
| active smoke/inferno | 158 | 0.645 | 0.341501 | 0.464319 | -0.122819 | 134 | 24 | 0.873418 | 0.651899 |
| recent utility last 5s | 21 | 0.086 | 0.524549 | 0.491588 | 0.032961 | 4 | 17 | 0.523810 | 0.523810 |
| flash effect present | 245 | 1.000 | 0.326257 | 0.450278 | -0.124021 | 209 | 36 | 0.914286 | 0.587755 |

## Active Smoke/Inferno Intervals

- `7.5s` - `56.5s`, rows `99`
- `84.0s` - `106.0s`, rows `45`
- `109.0s` - `115.5s`, rows `14`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `47.0`, LSTM `0.1958`, XGBoost `0.5124`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `47.5`, LSTM `0.2032`, XGBoost `0.5124`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `91.5`, LSTM `0.2031`, XGBoost `0.4989`, closer `lstm`, smoke `1`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `46.5`, LSTM `0.2215`, XGBoost `0.5124`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `48.5`, LSTM `0.2326`, XGBoost `0.5205`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `50.0`, LSTM `0.2326`, XGBoost `0.5203`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `45.0`, LSTM `0.2201`, XGBoost `0.5060`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `49.5`, LSTM `0.2350`, XGBoost `0.5200`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `49.0`, LSTM `0.2359`, XGBoost `0.5200`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `46.0`, LSTM `0.2253`, XGBoost `0.5091`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
